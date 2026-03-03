// AsgardBench Landing Page - Interactive Components

document.addEventListener('DOMContentLoaded', function() {

  // ─── Teaser Stepper ───
  const stepperTabs = document.querySelectorAll('.stepper-tab');
  const stepperPanels = document.querySelectorAll('.stepper-panel');
  const stepperDots = document.querySelectorAll('.stepper-dot');
  function activateStep(index) {
    stepperTabs.forEach(t => t.classList.remove('is-active'));
    stepperPanels.forEach(p => p.classList.remove('is-active'));
    stepperDots.forEach(d => d.classList.remove('is-active'));

    stepperTabs[index].classList.add('is-active');
    stepperPanels[index].classList.add('is-active');
    stepperDots[index].classList.add('is-active');
  }

  stepperTabs.forEach((tab, i) => {
    tab.addEventListener('click', () => activateStep(i));
  });

  stepperDots.forEach((dot, i) => {
    dot.addEventListener('click', () => activateStep(i));
  });

  // ─── Ablation Tabs ───
  const ablationTabLinks = document.querySelectorAll('.ablation-tab-link');
  const ablationContents = document.querySelectorAll('.ablation-tab-content');

  ablationTabLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.dataset.target;

      ablationTabLinks.forEach(l => l.parentElement.classList.remove('is-active'));
      link.parentElement.classList.add('is-active');

      ablationContents.forEach(c => c.classList.remove('is-active'));
      document.getElementById(target).classList.add('is-active');
    });
  });

  // ─── Copy BibTeX ───
  const copyBtn = document.querySelector('.copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const code = document.querySelector('.bibtex-block code').textContent;
      navigator.clipboard.writeText(code).then(() => {
        copyBtn.textContent = '✓ Copied!';
        setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
      });
    });
  }

  // ─── Smooth scroll for navbar ───
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ─── Bokeh Plan Tree – hide scrollbars & center x-range after load ───
  const treeIframe = document.getElementById('plan-tree-iframe');
  if (treeIframe) {
    treeIframe.addEventListener('load', function() {
      const poll = setInterval(function() {
        try {
          const iframeDoc = treeIframe.contentDocument || treeIframe.contentWindow.document;
          if (!iframeDoc) return;

          // Inject CSS to hide all scrollbars inside the Bokeh iframe
          // Bokeh 3.x uses shadow DOM, so we must inject CSS into each shadow root
          if (!iframeDoc.getElementById('hide-scrollbars')) {
            var scrollbarCSS = [
              '* { scrollbar-width: none !important; -ms-overflow-style: none !important; }',
              '*::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }',
            ].join('\n');

            // Light DOM style (html/body overflow + fallback)
            var style = iframeDoc.createElement('style');
            style.id = 'hide-scrollbars';
            style.textContent = scrollbarCSS + '\nhtml, body { overflow: hidden !important; }';
            iframeDoc.head.appendChild(style);

            // Inject into all existing shadow roots
            function injectIntoShadowRoots(root) {
              root.querySelectorAll('*').forEach(function(el) {
                if (el.shadowRoot) {
                  if (!el.shadowRoot.querySelector('.hide-sb-injected')) {
                    var s = iframeDoc.createElement('style');
                    s.className = 'hide-sb-injected';
                    s.textContent = scrollbarCSS;
                    el.shadowRoot.appendChild(s);
                  }
                  // Recurse into nested shadow roots
                  injectIntoShadowRoots(el.shadowRoot);
                }
              });
            }

            // Run injection repeatedly as Bokeh builds its DOM async
            var injectCount = 0;
            var injectInterval = setInterval(function() {
              injectIntoShadowRoots(iframeDoc);
              // Also force inline scrollbar-width on any overflow elements in light DOM
              iframeDoc.querySelectorAll('div[style*="overflow"]').forEach(function(el) {
                el.style.scrollbarWidth = 'none';
                el.style.msOverflowStyle = 'none';
              });
              injectCount++;
              if (injectCount > 20) clearInterval(injectInterval);
            }, 300);

            // MutationObserver to catch new shadow hosts
            new MutationObserver(function() {
              injectIntoShadowRoots(iframeDoc);
              iframeDoc.querySelectorAll('div[style*="overflow"]').forEach(function(el) {
                el.style.scrollbarWidth = 'none';
                el.style.msOverflowStyle = 'none';
              });
            }).observe(iframeDoc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] });
          }

          // Center the x-range
          const bk = treeIframe.contentWindow.Bokeh;
          if (!bk || !bk.documents || !bk.documents.length) return;
          const doc = bk.documents[0];
          const roots = doc.roots();
          if (!roots || !roots.length) return;

          const fig = roots[0];
          const xr = fig.x_range || (fig.children && fig.children[0] && fig.children[0].x_range);
          if (xr && typeof xr.start === 'number') {
            const fullStart = xr.reset_start != null ? xr.reset_start : xr.start;
            const fullEnd = xr.reset_end != null ? xr.reset_end : xr.end;
            const totalSpan = fullEnd - fullStart;
            const mid = (fullStart + fullEnd) / 2;
            const visibleSpan = totalSpan * 0.6;
            xr.start = mid - visibleSpan / 2;
            xr.end = mid + visibleSpan / 2;
            clearInterval(poll);
          }
        } catch(e) {
          // Not ready yet
        }
      }, 500);
      setTimeout(function() { clearInterval(poll); }, 10000);
    });
  }

});

function renderPlanTree(data, container, method) {
  const nodes = data.nodes;
  const edges = data.edges;

  const nodeById = {};
  nodes.forEach(n => { nodeById[n.id] = n; });

  // Flip y so first branch is at top
  const yMax = Math.max(...nodes.map(n => n.y));
  const yMin = Math.min(...nodes.map(n => n.y));
  const xMax = Math.max(...nodes.map(n => n.x));

  // Count children for each node
  const childrenCount = {};
  edges.forEach(e => {
    childrenCount[e.source] = (childrenCount[e.source] || 0) + 1;
  });

  // Helpers
  const flipY = n => yMax - n.y + yMin;

  const hoverlabelStyle = {
    bgcolor: '#fff',
    font: { family: 'Google Sans, Noto Sans, sans-serif', size: 12 },
    align: 'left',
  };

  // ─── Edge traces (dashed for collapsed chains) ───
  const solidEdgeX = [], solidEdgeY = [];
  const dashedEdgeX = [], dashedEdgeY = [];
  edges.forEach(e => {
    const src = nodeById[e.source];
    const tgt = nodeById[e.target];
    if (!src || !tgt) return;
    if (tgt.is_collapsed) {
      dashedEdgeX.push(src.x, tgt.x, null);
      dashedEdgeY.push(flipY(src), flipY(tgt), null);
    } else {
      solidEdgeX.push(src.x, tgt.x, null);
      solidEdgeY.push(flipY(src), flipY(tgt), null);
    }
  });

  const solidEdgeTrace = {
    x: solidEdgeX, y: solidEdgeY,
    mode: 'lines',
    line: { color: '#bbb', width: 1.5 },
    hoverinfo: 'none',
    showlegend: false,
  };

  const dashedEdgeTrace = {
    x: dashedEdgeX, y: dashedEdgeY,
    mode: 'lines',
    line: { color: '#bbb', width: 1.5, dash: 'dot' },
    hoverinfo: 'none',
    showlegend: false,
  };

  // ─── Categorize nodes ───
  const branchPoints = nodes.filter(n => (childrenCount[n.id] || 0) > 1);
  const branchIds = new Set(branchPoints.map(n => n.id));
  const collapsed = nodes.filter(n => n.is_collapsed && !branchIds.has(n.id));
  const terminals = nodes.filter(n => n.is_final);
  const regular = nodes.filter(n => !n.is_final && !n.is_collapsed && !branchIds.has(n.id));

  // ─── Hover builders ───
  function hoverRegular(n) {
    const nChildren = childrenCount[n.id] || 0;
    const lines = ['<b>' + n.chain_actions[0] + '</b>', ''];
    if (n.is_collapsed && n.chain_length > 1) {
      lines.push('(' + n.chain_length + ' steps: ' + n.chain_actions.join(' \u2192 ') + ')');
      lines.push('');
    }
    if (n.count > 1) lines.push(n.count + ' plans through here');
    if (nChildren > 1) lines.push('Branches into ' + nChildren + ' paths');
    return lines.join('<br>');
  }

  function hoverCollapsed(n) {
    const actions = n.chain_actions;
    const lines = ['<b>' + actions.length + ' steps collapsed</b>', ''];
    if (actions.length <= 8) {
      actions.forEach((a, i) => lines.push((i + 1) + '. ' + a));
    } else {
      for (let i = 0; i < 5; i++) lines.push((i + 1) + '. ' + actions[i]);
      lines.push('   \u22ee');
      for (let i = actions.length - 2; i < actions.length; i++) lines.push((i + 1) + '. ' + actions[i]);
    }
    return lines.join('<br>');
  }

  function hoverTerminal(n) {
    return '<b>' + n.chain_actions[0] + '</b><br><br>\u2713 End of plan';
  }

  // ─── Node traces ───
  const regularTrace = {
    x: regular.map(n => n.x),
    y: regular.map(n => flipY(n)),
    mode: 'markers',
    marker: { color: '#4A90D9', size: 7, line: { color: '#2a6ab5', width: 1 } },
    hovertext: regular.map(n => hoverRegular(n)),
    hoverinfo: 'text',
    hoverlabel: { ...hoverlabelStyle, bordercolor: '#2a6ab5' },
    name: 'Action',
    showlegend: true,
  };

  const branchTrace = {
    x: branchPoints.map(n => n.x),
    y: branchPoints.map(n => flipY(n)),
    mode: 'markers',
    marker: { color: '#E67E22', size: 11, symbol: 'diamond', line: { color: '#a85a00', width: 1.5 } },
    hovertext: branchPoints.map(n => hoverRegular(n)),
    hoverinfo: 'text',
    hoverlabel: { ...hoverlabelStyle, bordercolor: '#a85a00' },
    name: 'Branch point',
    showlegend: true,
  };

  const collapsedTrace = {
    x: collapsed.map(n => n.x),
    y: collapsed.map(n => flipY(n)),
    mode: 'markers',
    marker: {
      color: 'rgba(142, 68, 173, 0.15)',
      size: collapsed.map(n => Math.min(28, 12 + Math.log2(n.chain_length) * 5)),
      symbol: 'circle',
      line: { color: '#8E44AD', width: 2 },
    },
    hovertext: collapsed.map(n => hoverCollapsed(n)),
    hoverinfo: 'text',
    hoverlabel: { ...hoverlabelStyle, bordercolor: '#8E44AD' },
    name: 'Collapsed steps',
    showlegend: true,
  };

  const finalTrace = {
    x: terminals.map(n => n.x),
    y: terminals.map(n => flipY(n)),
    mode: 'markers',
    marker: { color: '#2ECC71', size: 7, symbol: 'square', line: { color: '#1a9c54', width: 1 } },
    hovertext: terminals.map(n => hoverTerminal(n)),
    hoverinfo: 'text',
    hoverlabel: { ...hoverlabelStyle, bordercolor: '#1a9c54' },
    name: 'End of plan',
    showlegend: true,
  };

  // ─── Labels for branch points only (rest visible on hover) ───
  const labeled = branchPoints;
  const labelTrace = {
    x: labeled.map(n => n.x),
    y: labeled.map(n => flipY(n)),
    mode: 'text',
    text: labeled.map(n => n.chain_actions ? n.chain_actions[0] : n.action_desc),
    textposition: 'top center',
    textfont: { family: 'Google Sans, Noto Sans, sans-serif', size: 8, color: '#444' },
    hoverinfo: 'none',
    showlegend: false,
  };

  // ─── Collapsed chain count labels ───
  const collapsedLabelTrace = {
    x: collapsed.map(n => n.x),
    y: collapsed.map(n => flipY(n)),
    mode: 'text',
    text: collapsed.map(n => n.chain_length + ''),
    textfont: { family: 'Google Sans, Noto Sans, sans-serif', size: 8, color: '#8E44AD' },
    hoverinfo: 'none',
    showlegend: false,
  };

  const layout = {
    xaxis: {
      showgrid: false, zeroline: false, showticklabels: false,
      range: [-1.0, xMax + 1.0],
      fixedrange: true,
    },
    yaxis: {
      showgrid: false, zeroline: false, showticklabels: false,
      range: [yMin - 1.0, yMax - yMin + yMin + 1.0],
      fixedrange: true,
    },
    margin: { l: 10, r: 30, t: 30, b: 10 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    hovermode: 'closest',
    dragmode: false,
    legend: {
      orientation: 'h',
      x: 0.5, xanchor: 'center',
      y: 1.08,
      font: { family: 'Google Sans, Noto Sans, sans-serif', size: 11 },
    },
    annotations: [{
      x: -0.5, y: flipY(nodes[0]),
      xref: 'x', yref: 'y',
      text: '<b>"Prepare coffee"</b>',
      showarrow: true, arrowhead: 2,
      ax: -55, ay: 0,
      font: { family: 'Google Sans, sans-serif', size: 12, color: '#333' },
    }],
  };

  const config = {
    responsive: true,
    staticPlot: false,
    displayModeBar: false,
    scrollZoom: false,
  };

  const traces = [solidEdgeTrace, dashedEdgeTrace, collapsedTrace, regularTrace, branchTrace, finalTrace, collapsedLabelTrace, labelTrace];

  if (method === 'react') {
    Plotly.react(container, traces, layout, config);
  } else {
    Plotly.newPlot(container, traces, layout, config);
  }
}
