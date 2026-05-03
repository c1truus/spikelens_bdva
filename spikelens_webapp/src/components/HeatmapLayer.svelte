<script>
  import { onMount, afterUpdate } from 'svelte';
  import * as d3 from 'd3';
  import { getLayerValues, getSpikeMask, arrayMinMax } from '../lib/traceUtils.js';

  export let trace;
  export let layer = 'fc1';
  export let t = 0;
  export let mode = 'membrane';
  export let title = '';
  export let rows = 32;
  export let cols = 32;
  export let size = 230;
  export let output = false;

  let canvas;
  let hover = null;

  $: values = getLayerValues(trace, layer, t, mode);
  $: spikeMask = getSpikeMask(trace, layer, t);
  $: domainSource = mode === 'membrane' && trace?.membrane_dense?.[layer]
    ? trace.membrane_dense[layer]
    : values;
  $: [vmin, vmax] = arrayMinMax(domainSource);

  function colorValue(v) {
    if (mode === 'spike') {
      return v ? '#ffd84d' : '#111827';
    }
    if (mode === 'cumulative') {
      const scale = d3.scaleSequential(d3.interpolateMagma).domain([0, Math.max(1, vmax)]);
      return scale(v);
    }
    const scale = d3.scaleSequential(d3.interpolateViridis).domain([vmin, vmax]);
    return scale(v);
  }

  function draw() {
    if (!canvas || !values?.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,size,size);

    const cellW = size / cols;
    const cellH = size / rows;

    for (let i = 0; i < rows*cols; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const v = values[i] ?? 0;
      ctx.fillStyle = colorValue(v);
      ctx.fillRect(c*cellW, r*cellH, Math.ceil(cellW), Math.ceil(cellH));

      if (mode !== 'spike' && spikeMask?.[i]) {
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = Math.max(1, cellW * 0.14);
        ctx.strokeRect(c*cellW+1, r*cellH+1, Math.max(1, cellW-2), Math.max(1, cellH-2));
      }
    }
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.floor(x / (rect.width / cols));
    const r = Math.floor(y / (rect.height / rows));
    const idx = r*cols+c;
    if (idx >= 0 && idx < rows*cols) {
      hover = {
        x: e.clientX - rect.left + 12,
        y: e.clientY - rect.top + 12,
        idx,
        row: r,
        col: c,
        value: values[idx] ?? 0,
        spike: spikeMask?.[idx] ?? 0
      };
    }
  }

  function onLeave() { hover = null; }

  onMount(draw);
  afterUpdate(draw);
</script>

<div class="layer-wrap">
  <div class="layer-title">{title}</div>
  <div class="canvas-wrap" on:mousemove={onMove} on:mouseleave={onLeave}>
    <canvas bind:this={canvas} class="heatmap"></canvas>
    {#if hover}
      <div class="tooltip" style={`left:${hover.x}px;top:${hover.y}px`}>
        <div><b>{layer}</b> neuron {hover.idx}</div>
        <div>row={hover.row}, col={hover.col}</div>
        <div>value={Number(hover.value).toFixed(3)}</div>
        <div>spike={hover.spike}</div>
      </div>
    {/if}
  </div>
</div>
