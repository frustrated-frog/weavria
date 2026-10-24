(function exposeMarkdown(root, factory) {
  const markdown = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = markdown;
  }
  if (root) {
    root.WeavriaMarkdown = markdown;
  }
})(typeof globalThis === 'undefined' ? this : globalThis, function createMarkdown() {
  'use strict';

  const inlinePattern = /(`([^`\n]+)`|\*\*([^*\n]+)\*\*|__([^_\n]+)__|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|\*([^*\n]+)\*|_([^_\n]+)_)/g;

  function parseInline(value) {
    const source = String(value ?? '');
    const result = [];
    let offset = 0;
    inlinePattern.lastIndex = 0;
    for (const match of source.matchAll(inlinePattern)) {
      if (match.index > offset) {
        result.push({ type: 'text', text: source.slice(offset, match.index) });
      }
      if (match[2] !== undefined) {
        result.push({ type: 'code', text: match[2] });
      } else if (match[3] !== undefined || match[4] !== undefined) {
        result.push({ type: 'strong', text: match[3] ?? match[4] });
      } else if (match[5] !== undefined) {
        result.push({ type: 'link', text: match[5], href: match[6] });
      } else {
        result.push({ type: 'emphasis', text: match[7] ?? match[8] });
      }
      offset = match.index + match[0].length;
    }
    if (offset < source.length) {
      result.push({ type: 'text', text: source.slice(offset) });
    }
    return result;
  }

  function splitTableRow(value) {
    let source = String(value ?? '').trim();
    if (!source.includes('|')) return null;
    if (source.startsWith('|')) source = source.slice(1);
    if (source.endsWith('|')) source = source.slice(0, -1);

    const cells = [];
    let cell = '';
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (character === '\\' && source[index + 1] === '|') {
        cell += '|';
        index += 1;
      } else if (character === '|') {
        cells.push(cell.trim());
        cell = '';
      } else {
        cell += character;
      }
    }
    cells.push(cell.trim());
    return cells.length > 1 ? cells : null;
  }

  function parseTable(value, delimiter) {
    const headers = splitTableRow(value);
    const delimiters = splitTableRow(delimiter);
    if (!headers || !delimiters || headers.length !== delimiters.length) return null;

    const alignments = delimiters.map((cell) => {
      const marker = cell.trim();
      if (!/^:?-{3,}:?$/.test(marker)) return null;
      if (marker.startsWith(':') && marker.endsWith(':')) return 'center';
      if (marker.endsWith(':')) return 'right';
      return 'left';
    });
    if (alignments.includes(null)) return null;
    return { headers: headers.map(parseInline), alignments };
  }

  function parseMarkdown(value) {
    const lines = String(value ?? '').replace(/\r\n?/g, '\n').split('\n');
    const blocks = [];
    let paragraph = [];
    let list = null;
    let quote = [];
    let code = null;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      blocks.push({ type: 'paragraph', lines: paragraph.map(parseInline) });
      paragraph = [];
    };
    const flushList = () => {
      if (!list) return;
      blocks.push({ type: list.type, items: list.items.map(parseInline) });
      list = null;
    };
    const flushQuote = () => {
      if (!quote.length) return;
      blocks.push({ type: 'blockquote', lines: quote.map(parseInline) });
      quote = [];
    };
    const flushOpenBlocks = () => {
      flushParagraph();
      flushList();
      flushQuote();
    };

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex];
      if (code !== null) {
        if (/^ {0,3}```\s*$/.test(line)) {
          blocks.push({ type: 'code_block', text: code.join('\n') });
          code = null;
        } else {
          code.push(line);
        }
        continue;
      }

      if (/^ {0,3}```(?:[A-Za-z0-9_-]+)?\s*$/.test(line)) {
        flushOpenBlocks();
        code = [];
        continue;
      }
      if (!line.trim()) {
        flushOpenBlocks();
        continue;
      }

      const table = lineIndex + 1 < lines.length
        ? parseTable(line, lines[lineIndex + 1])
        : null;
      if (table) {
        flushOpenBlocks();
        const rows = [];
        lineIndex += 2;
        while (lineIndex < lines.length) {
          const cells = splitTableRow(lines[lineIndex]);
          if (!cells || cells.length !== table.headers.length) break;
          rows.push(cells.map(parseInline));
          lineIndex += 1;
        }
        blocks.push({
          type: 'table',
          headers: table.headers,
          alignments: table.alignments,
          rows
        });
        lineIndex -= 1;
        continue;
      }

      const heading = line.match(/^ {0,3}(#{1,4})\s+(.+?)\s*#*$/);
      if (heading) {
        flushOpenBlocks();
        blocks.push({
          type: 'heading',
          level: heading[1].length,
          content: parseInline(heading[2])
        });
        continue;
      }

      if (/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        flushOpenBlocks();
        blocks.push({ type: 'divider' });
        continue;
      }

      const unordered = line.match(/^ {0,3}[-+*]\s+(.+)$/);
      const ordered = line.match(/^ {0,3}\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        flushParagraph();
        flushQuote();
        const type = unordered ? 'unordered_list' : 'ordered_list';
        if (list && list.type !== type) flushList();
        if (!list) list = { type, items: [] };
        list.items.push((unordered ?? ordered)[1]);
        continue;
      }

      const quoted = line.match(/^ {0,3}>\s?(.*)$/);
      if (quoted) {
        flushParagraph();
        flushList();
        quote.push(quoted[1]);
        continue;
      }

      flushList();
      flushQuote();
      paragraph.push(line.trim());
    }

    if (code !== null) {
      blocks.push({ type: 'code_block', text: code.join('\n') });
    }
    flushOpenBlocks();
    return blocks;
  }

  function appendInline(parent, nodes) {
    nodes.forEach((node) => {
      if (node.type === 'text') {
        parent.append(document.createTextNode(node.text));
        return;
      }
      const tag = ({ strong: 'strong', emphasis: 'em', code: 'code', link: 'a' })[node.type];
      const element = document.createElement(tag);
      element.textContent = node.text;
      if (node.type === 'link') {
        element.href = node.href;
        element.target = '_blank';
        element.rel = 'noopener noreferrer';
        element.referrerPolicy = 'no-referrer';
      }
      parent.append(element);
    });
  }

  function appendLines(parent, lines) {
    lines.forEach((line, index) => {
      if (index) parent.append(document.createElement('br'));
      appendInline(parent, line);
    });
  }

  function appendTableCell(row, tag, content, alignment) {
    const cell = document.createElement(tag);
    cell.className = `align-${alignment}`;
    if (tag === 'th') cell.scope = 'col';
    appendInline(cell, content);
    row.append(cell);
  }

  function renderMarkdown(container, value) {
    const fragment = document.createDocumentFragment();
    parseMarkdown(value).forEach((block) => {
      let element;
      if (block.type === 'heading') {
        element = document.createElement(`h${Math.min(block.level + 1, 6)}`);
        appendInline(element, block.content);
      } else if (block.type === 'unordered_list' || block.type === 'ordered_list') {
        element = document.createElement(block.type === 'unordered_list' ? 'ul' : 'ol');
        block.items.forEach((item) => {
          const entry = document.createElement('li');
          appendInline(entry, item);
          element.append(entry);
        });
      } else if (block.type === 'table') {
        element = document.createElement('div');
        element.className = 'table-scroll';
        const table = document.createElement('table');
        const head = document.createElement('thead');
        const headerRow = document.createElement('tr');
        block.headers.forEach((header, index) => {
          appendTableCell(headerRow, 'th', header, block.alignments[index]);
        });
        head.append(headerRow);
        table.append(head);
        const body = document.createElement('tbody');
        block.rows.forEach((cells) => {
          const row = document.createElement('tr');
          cells.forEach((cell, index) => {
            appendTableCell(row, 'td', cell, block.alignments[index]);
          });
          body.append(row);
        });
        table.append(body);
        element.append(table);
      } else if (block.type === 'blockquote') {
        element = document.createElement('blockquote');
        appendLines(element, block.lines);
      } else if (block.type === 'code_block') {
        element = document.createElement('pre');
        const codeElement = document.createElement('code');
        codeElement.textContent = block.text;
        element.append(codeElement);
      } else if (block.type === 'divider') {
        element = document.createElement('hr');
      } else {
        element = document.createElement('p');
        appendLines(element, block.lines);
      }
      fragment.append(element);
    });
    container.replaceChildren(fragment);
  }

  return Object.freeze({ parseInline, parseMarkdown, renderMarkdown });
});
