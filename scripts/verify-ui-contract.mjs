import { existsSync } from 'node:fs'
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const sourceExtensions = new Set(['.css', '.ts', '.tsx'])
const forbiddenTokens = ['violet', 'cyan', 'hero-card', 'eyebrow', 'metric-card', 'settings-card']

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map(async (entry) => {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) return sourceFiles(target)
    return sourceExtensions.has(path.extname(entry.name)) ? [target] : []
  }))
  return nested.flat()
}

export async function verifyUiContract(root = process.cwd()) {
  const sourceRoot = path.join(root, 'src')
  const files = await sourceFiles(sourceRoot)
  const errors = []

  for (const file of files) {
    const relative = path.relative(root, file).replaceAll('\\', '/')
    const source = await readFile(file, 'utf8')
    const lower = source.toLowerCase()

    for (const token of forbiddenTokens) {
      if (lower.includes(token)) errors.push(`${relative}: 禁止 token ${token}`)
    }
    if (/font-family\s*:[^;]*(?:Georgia|Times New Roman|(?<!sans-)\bserif\b)/i.test(source)) {
      errors.push(`${relative}: 禁止 serif 字体`)
    }
    for (const match of source.matchAll(/border-radius\s*:\s*(\d+(?:\.\d+)?)px/gi)) {
      if (Number(match[1]) > 10) errors.push(`${relative}: 圆角 ${match[1]}px 超过 10px`)
    }
  }

  const styles = await readFile(path.join(sourceRoot, 'styles.css'), 'utf8')
  const baseSize = styles.match(/html\s*\{[^}]*font-size\s*:\s*calc\((\d+(?:\.\d+)?)px/si)
  if (!baseSize || Number(baseSize[1]) < 14) errors.push('src/styles.css: 基础字号必须至少 14px')
  if (existsSync(path.join(sourceRoot, 'components', 'ui', 'card.tsx'))) {
    errors.push('src/components/ui/card.tsx: 禁止 Card 原语')
  }

  return errors
}

const isCli = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isCli) {
  const errors = await verifyUiContract()
  if (errors.length) {
    console.error(errors.join('\n'))
    process.exitCode = 1
  } else {
    console.log('UI_CONTRACT_OK')
  }
}
