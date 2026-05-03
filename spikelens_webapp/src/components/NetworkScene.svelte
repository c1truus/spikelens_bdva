<script>
  import { tick } from 'svelte';

  export let trace;
  export let t = 0;
  export let mode = 'spike';

  let inputCanvas;
  let fc1Canvas;
  let fc2Canvas;
  let outCanvas;

  const layerMeta = {
    input: { rows: 28, cols: 28, width: 150, height: 150 },
    fc1: { rows: 32, cols: 32, width: 220, height: 220 },
    fc2: { rows: 16, cols: 16, width: 160, height: 160 },
    out: { rows: 1, cols: 10, width: 300, height: 44 }
  };

  $: redrawTrigger = `${trace?.model_id ?? 'none'}-${trace?.sample_idx ?? 'none'}-${trace?.pred_label ?? 'none'}-${t}-${mode}`;

  $: if (trace && redrawTrigger) {
    redraw();
  }

  async function redraw() {
    await tick();
    drawLayer(inputCanvas, 'input');
    drawLayer(fc1Canvas, 'fc1');
    drawLayer(fc2Canvas, 'fc2');
    drawLayer(outCanvas, 'out');
  }

  function getDenseSpikes(layer) {
    return trace?.spikes_dense?.[layer] ?? trace?.spikes?.[layer] ?? [];
  }

  function getMembrane(layer) {
    return trace?.membrane?.[layer] ?? [];
  }

  function getFrame(layer) {
    const step = Number(t);

    if (mode === 'membrane' && layer !== 'input') {
      const mem = getMembrane(layer);
      if (mem?.[step]) return mem[step];
    }

    if (mode === 'cumulative') {
      const spikes = getDenseSpikes(layer);
      const size =
        layer === 'input' ? 784 :
        layer === 'fc1' ? 1024 :
        layer === 'fc2' ? 256 :
        10;

      const accum = Array(size).fill(0);
      for (let i = 0; i <= step; i++) {
        const frame = spikes?.[i];
        if (!frame) continue;
        for (let n = 0; n < Math.min(size, frame.length); n++) {
          accum[n] += Number(frame[n] ?? 0);
        }
      }
      return accum;
    }

    const spikes = getDenseSpikes(layer);
    return spikes?.[step] ?? [];
  }

  function drawLayer(canvas, layer) {
    if (!canvas || !trace) return;

    const meta = layerMeta[layer];
    const values = getFrame(layer);
    const spikes = getDenseSpikes(layer)?.[Number(t)] ?? [];

    const dpr = window.devicePixelRatio || 1;
    const cssW = meta.width;
    const cssH = meta.height;

    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const rows = meta.rows;
    const cols = meta.cols;
    const cellW = cssW / cols;
    const cellH = cssH / rows;

    const numericValues = Array.from({ length: rows * cols }, (_, i) => Number(values?.[i] ?? 0));
    const minV = Math.min(...numericValues);
    const maxV = Math.max(...numericValues);

    ctx.fillStyle = '#020617';
    roundRect(ctx, 0, 0, cssW, cssH, 14);
    ctx.fill();

    for (let i = 0; i < rows * cols; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const v = Number(values?.[i] ?? 0);
      const spiked = Number(spikes?.[i] ?? 0) > 0;

      let fill;
      if (mode === 'membrane' && layer !== 'input') {
        // True grayscale: black = low membrane, white = high membrane.
        fill = grayscaleColor(v, minV, maxV);
      } else if (mode === 'cumulative') {
        fill = cumulativeColor(v, maxV);
      } else {
        fill = spiked || v > 0 ? '#ffd84d' : '#06101f';
      }

      ctx.fillStyle = fill;
      ctx.fillRect(
        c * cellW + 0.35,
        r * cellH + 0.35,
        Math.max(0.5, cellW - 0.7),
        Math.max(0.5, cellH - 0.7)
      );

      // In grayscale membrane mode, keep spike events visible with yellow outlines.
      if (mode === 'membrane' && spiked) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = layer === 'out' ? 2.5 : 1.05;
        ctx.strokeRect(
          c * cellW + 1.05,
          r * cellH + 1.05,
          Math.max(1, cellW - 2.1),
          Math.max(1, cellH - 2.1)
        );
      }
    }

    ctx.strokeStyle = 'rgba(226, 232, 240, 0.28)';
    ctx.lineWidth = 1;
    roundRect(ctx, 0.5, 0.5, cssW - 1, cssH - 1, 14);
    ctx.stroke();
  }

  function grayscaleColor(v, minV, maxV) {
    const span = Math.max(maxV - minV, 1e-9);
    const x = Math.max(0, Math.min(1, (v - minV) / span));
    // Slight floor keeps grid visible on dark background.
    const g = Math.round(8 + x * 247);
    return `rgb(${g}, ${g}, ${g})`;
  }

  function cumulativeColor(v, maxV) {
    if (v <= 0) return '#06101f';
    const x = Math.min(1, v / Math.max(maxV, 1));
    const r = Math.round(35 + x * 220);
    const g = Math.round(70 + x * 185);
    const b = Math.round(95 - x * 65);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + radius, y, radius);
    ctx.closePath();
  }
</script>

<div class="network-scene spikelens-scene scene-v3">
  <svg class="connectors spikelens-connectors" viewBox="0 0 1000 430" preserveAspectRatio="none" aria-hidden="true">
    <defs>
      <linearGradient id="pulse-main-v3" x1="0" x2="1">
        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.10" />
        <stop offset="52%" stop-color="#facc15" stop-opacity="0.46" />
        <stop offset="100%" stop-color="#a78bfa" stop-opacity="0.14" />
      </linearGradient>
    </defs>

    <path d="M165 185 C265 105 350 100 430 150" stroke="url(#pulse-main-v3)" stroke-width="14" fill="none" />
    <path d="M560 150 C620 75 705 82 790 158" stroke="url(#pulse-main-v3)" stroke-width="14" fill="none" />
    <path d="M780 250 C705 328 595 348 505 356" stroke="url(#pulse-main-v3)" stroke-width="12" fill="none" />
  </svg>

  <div class="layer layer-input">
    <div class="layer-title">Input 28×28</div>
    <canvas bind:this={inputCanvas} class="heatmap"></canvas>
  </div>

  <div class="layer layer-fc1">
    <div class="layer-title">FC1 32×32</div>
    <canvas bind:this={fc1Canvas} class="heatmap"></canvas>
  </div>

  <div class="layer layer-fc2">
    <div class="layer-title">FC2 16×16</div>
    <canvas bind:this={fc2Canvas} class="heatmap"></canvas>
  </div>

  <div class="layer layer-output-bottom">
    <div class="layer-title">Output classes 0–9</div>
    <canvas bind:this={outCanvas} class="heatmap output-strip"></canvas>
    <div class="class-labels" aria-hidden="true">
      {#each Array(10) as _, i}
        <span>{i}</span>
      {/each}
    </div>
  </div>
</div>