"""Build offline HTML source documentation: python -m tools.build_docs."""
import ast
from html import escape
from pathlib import Path
import subprocess
import sys
from tools.notebook_contract import definition_tree

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Document package/tools via pdoc and notebook functions from their actual AST."""
    output = ROOT / 'docs/html'
    subprocess.run([sys.executable, '-m', 'pdoc', 'mma3001_f1',
                    'tools.notebook_contract', 'tools.benchmark', 'tools.reproduce', '-o', str(output)], cwd=ROOT, check=True)
    tree = definition_tree(ROOT / 'MMA3001_F1_Next_Lap_Prediction.ipynb')
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Notebook API</title>',
             '<style>body{max-width:1000px;margin:40px auto;font:16px/1.5 system-ui;padding:20px}pre{overflow:auto;background:#f3f5f8;padding:16px}a{color:#176b9a}</style>',
             '<h1>Assessed notebook source documentation</h1>',
             '<p>Generated from the current notebook, without executing training or download cells. The active model is implemented in the notebook; mma3001_f1.modeling is an earlier optional scaffold.</p>',
             '<p><a href="index.html">Package and tools API</a></p>']
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            parts.append(f'<h2 id="{node.name}">{node.name}</h2><p>{escape(ast.get_docstring(node) or "No docstring provided")}</p><pre>{escape(ast.unparse(node))}</pre>')
    for relative in ('dataset/fastf1_export_dataset.py', 'dataset/fastf1_export_multi_gp.py'):
        parts.append(f'<h1>{escape(relative)}</h1>')
        exporter = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
        for node in exporter.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                parts.append(f'<h2>{escape(node.name)}</h2><p>{escape(ast.get_docstring(node) or "No docstring provided")}</p><pre>{escape(ast.unparse(node))}</pre>')
    parts.append('</html>')
    (output / 'notebook.html').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()
