#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function readSkillMeta(skillDir) {
  const file = path.join(skillDir, 'SKILL.md');
  if (!fs.existsSync(file)) return null;
  const content = fs.readFileSync(file, 'utf8');
  // try to parse YAML frontmatter name & description
  const fmMatch = content.match(/^---\n([\s\S]*?)\n---/);
  let name = null;
  let description = null;
  if (fmMatch) {
    const fm = fmMatch[1];
    const nameMatch = fm.match(/^name:\s*(.+)$/m);
    const descMatch = fm.match(/^description:\s*(.+)$/m);
    if (nameMatch) name = nameMatch[1].trim();
    if (descMatch) description = descMatch[1].trim();
  }
  // fallback: use first heading and first paragraph
  if (!name) {
    const hMatch = content.match(/^#\s*(.+)$/m);
    if (hMatch) name = hMatch[1].trim();
  }
  if (!description) {
    const para = content.split('\n\n').find(p => p && !p.startsWith('#') && !p.startsWith('---')) || '';
    description = para.replace(/\n/g, ' ').trim().slice(0, 300);
  }
  return { name, description, file };
}

function score(meta, tokens) {
  if (!meta) return 0;
  const text = ((meta.name || '') + ' ' + (meta.description || '')).toLowerCase();
  let s = 0;
  for (const t of tokens) {
    if (!t) continue;
    if (text.includes(t)) s += text.split(t).length - 1;
  }
  return s;
}

function findSkills(query) {
  const skillsRoot = path.join(process.cwd(), '.agents', 'skills');
  if (!fs.existsSync(skillsRoot)) {
    console.error('No skills directory found at', skillsRoot);
    process.exit(2);
  }
  const dirs = fs.readdirSync(skillsRoot, { withFileTypes: true }).filter(d => d.isDirectory()).map(d=>path.join(skillsRoot,d.name));
  const tokens = query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const results = [];
  for (const d of dirs) {
    const meta = readSkillMeta(d);
    if (!meta) continue;
    const s = score(meta, tokens);
    results.push({ dir: d, score: s, meta });
  }
  results.sort((a,b)=>{
    const diff = b.score - a.score;
    if (diff !== 0) return diff;
    const an = (a.meta && a.meta.name) ? a.meta.name : '';
    const bn = (b.meta && b.meta.name) ? b.meta.name : '';
    return an.localeCompare(bn);
  });
  return results;
}

function printResults(results, query) {
  console.log(`Query: "${query}"`);
  if (!results.length) { console.log('No skills found.'); return; }
  const top = results[0];
  if (top.score === 0) console.log('No strong match found. Top candidates:');
  console.log('\nTop matches:');
  results.slice(0,10).forEach(r=>{
    console.log(`- ${r.meta.name} (score=${r.score}) -> ${r.meta.file}`);
    console.log(`  ${r.meta.description}`);
  });
}

function main() {
  const args = process.argv.slice(2);
  if (!args.length) {
    console.log('Usage: node scripts/select-skill.js "your request text"');
    process.exit(1);
  }
  const query = args.join(' ');
  const results = findSkills(query);
  printResults(results, query);
}

main();
