<script>
  import { onDestroy, tick } from 'svelte';

  export let trace;

  let originalCanvas;
  let firstSpikeCanvas;
  let currentSpikeCanvas;
  let cumulativeCanvas;

  let demoT = 0;
  let demoPlaying = false;
  let demoSpeed = 2;
  let timer = null;

  $: numSteps = Number(trace?.num_steps ?? trace?.timesteps ?? 25);
  $: maxT = Math.max(0, numSteps - 1);
  $: inputSpikes = trace?.spikes_dense?.input ?? trace?.spikes?.input ?? [];
  $: originalImage = flattenImage(trace?.original_image);
  $: currentFrame = inputSpikes?.[Number(demoT)] ?? [];
  $: cumulativeFrame = buildCumulativeFrame(inputSpikes, Number(demoT));
  $: firstSpikeTimes = buildFirstSpikeTimes(inputSpikes, numSteps);
  $: spikeCounts = inputSpikes.map((frame) => countSpikes(frame));
  $: totalInputSpikes = spikeCounts.reduce((a, b) => a + b, 0);
  $: activePixels = firstSpikeTimes.filter((x) => x >= 0).length;
  $: earliestSpike = Math.min(...firstSpikeTimes.filter((x) => x >= 0), Infinity);
  $: latestSpike = Math.max(...firstSpikeTimes.filter((x) => x >= 0), -Infinity);

  $: if (trace) {
    if (demoT > maxT) demoT = 0;
    redrawAll();
  }

  $: {
    stopTimer();
    if (demoPlaying && trace) {
      timer = setInterval(() => {
        demoT = Number(demoT) >= maxT ? 0 : Number(demoT) + 1;
      }, 1000 / Number(demoSpeed));
    }
  }

  async function redrawAll() {
    await tick();
    drawOriginal();
    drawFirstSpikeMap();
    drawBinaryMap(currentSpikeCanvas, currentFrame, '#ffd84d');
    drawBinaryMap(cumulativeCanvas, cumulativeFrame, '#e5e7eb');
  }

  function flattenImage(img) {
    if (!img) return Array(784).fill(0);
    if (Array.isArray(img) && Array.isArray(img[0])) return img.flat().map(Number);
    if (Array.isArray(img)) return img.map(Number);
    return Array(784).fill(0);
  }

  function countSpikes(frame) {
    if (!Array.isArray(frame)) return 0;
    let c = 0;
    for (const v of frame) c += Number(v ?? 0) > 0 ? 1 : 0;
    return c;
  }

  function buildCumulativeFrame(spikes, t) {
    const out = Array(784).fill(0);
    for (let step = 0; step <= t; step++) {
      const frame = spikes?.[step];
      if (!frame) continue;
      for (let i = 0; i < Math.min(784, frame.length); i++) {
        if (Number(frame[i]) > 0) out[i] = 1;
      }
    }
    return out;
  }

  function buildFirstSpikeTimes(spikes, steps) {
    const out = Array(784).fill(-1);
    for (let step = 0; step < steps; step++) {
      const frame = spikes?.[step];
      if (!frame) continue;
      for (let i = 0; i < Math.min(784, frame.length); i++) {
        if (out[i] < 0 && Number(frame[i]) > 0) out[i] = step;
      }
    }
    return out;
  }

  function setupCanvas(canvas, cssSize = 196) {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(cssSize * dpr);
    canvas.height = Math.floor(cssSize * dpr);
    canvas.style.width = `${cssSize}px`;
    canvas.style.height = `${cssSize}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssSize, cssSize);
    return { ctx, cssSize };
  }

  function drawOriginal() {
    const s = setupCanvas(originalCanvas);
    if (!s) return;
    const { ctx, cssSize } = s;
    const cell = cssSize / 28;
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, cssSize, cssSize);
    const maxVal = Math.max(...originalImage, 1);
    for (let i = 0; i < 784; i++) {
      const x = i % 28;
      const y = Math.floor(i / 28);
      const v = Math.max(0, Math.min(1, Number(originalImage[i] ?? 0) / maxVal));
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g}, ${g}, ${g})`;
      ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }

  function drawFirstSpikeMap() {
    const s = setupCanvas(firstSpikeCanvas);
    if (!s) return;
    const { ctx, cssSize } = s;
    const cell = cssSize / 28;
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, cssSize, cssSize);
    for (let i = 0; i < 784; i++) {
      const t0 = firstSpikeTimes[i];
      const x = i % 28;
      const y = Math.floor(i / 28);
      if (t0 < 0) {
        ctx.fillStyle = '#030712';
      } else {
        const early = 1 - t0 / Math.max(1, maxT);
        const r = Math.round(55 + early * 200);
        const g = Math.round(55 + early * 200);
        const b = Math.round(55 + early * 130);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      }
      ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }

  function drawBinaryMap(canvas, frame, color) {
    const s = setupCanvas(canvas);
    if (!s) return;
    const { ctx, cssSize } = s;
    const cell = cssSize / 28;
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, cssSize, cssSize);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.055)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 28; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, cssSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(cssSize, i * cell);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (let i = 0; i < 784; i++) {
      if (Number(frame?.[i] ?? 0) <= 0) continue;
      const x = i % 28;
      const y = Math.floor(i / 28);
      ctx.fillRect(x * cell + 0.45, y * cell + 0.45, Math.max(1, cell - 0.9), Math.max(1, cell - 0.9));
    }
  }

  function jumpTo(step) {
    demoT = Math.max(0, Math.min(maxT, Number(step)));
  }

  function toggleDemo() {
    demoPlaying = !demoPlaying;
  }

  function stopTimer() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  onDestroy(stopTimer);
</script>

<section class="latency-explainer card wide editorial-section">
  <div class="section-kicker">Input encoding</div>
  <div class="latency-head editorial-head">
    <div>
      <h2>From image to spike train</h2>
      <p class="latency-lead editorial-lead">
        The input digit is not passed as one static image. It is converted into a short event sequence:
        high-intensity pixels fire earlier, weak pixels fire later, and inactive pixels remain silent.
      </p>
    </div>

    <div class="latency-metrics compact-metrics">
      <div><strong>{numSteps}</strong><span>frames</span></div>
      <div><strong>{activePixels}</strong><span>active pixels</span></div>
      <div><strong>{totalInputSpikes}</strong><span>events</span></div>
      <div><strong>{earliestSpike === Infinity ? '—' : `${earliestSpike}–${latestSpike}`}</strong><span>spike window</span></div>
    </div>
  </div>

  <div class="latency-controls editorial-controls">
    <div class="control-group">
      <label for="latency-demo-slider">Encoding frame t={demoT}</label>
      <input id="latency-demo-slider" type="range" min="0" max={maxT} bind:value={demoT} />
    </div>

    <div class="play-row latency-play-row">
      <button type="button" class="primary" on:click={toggleDemo}>{demoPlaying ? 'Pause' : 'Play'}</button>
      <select aria-label="Latency encoding speed" bind:value={demoSpeed}>
        <option value={1}>1 step/s</option>
        <option value={2}>2 step/s</option>
        <option value={4}>4 step/s</option>
        <option value={12}>12 step/s</option>
      </select>
    </div>
  </div>

  <div class="latency-visual-grid editorial-visual-grid">
    <article class="encoding-card">
      <h3>Source image</h3>
      <canvas bind:this={originalCanvas} class="encoding-canvas"></canvas>
      <p>Pixel intensity is the original evidence.</p>
    </article>

    <article class="encoding-card">
      <h3>First-spike map</h3>
      <canvas bind:this={firstSpikeCanvas} class="encoding-canvas"></canvas>
      <p>Brighter cells fired earlier in the sequence.</p>
    </article>

    <article class="encoding-card active-card">
      <h3>Frame t={demoT}</h3>
      <canvas bind:this={currentSpikeCanvas} class="encoding-canvas"></canvas>
      <p>Yellow cells are pixels firing at this frame.</p>
    </article>

    <article class="encoding-card">
      <h3>Accumulated input</h3>
      <canvas bind:this={cumulativeCanvas} class="encoding-canvas"></canvas>
      <p>White cells have fired at least once so far.</p>
    </article>
  </div>

  <div class="latency-timeline-card compact-timeline-card">
    <div class="timeline-title-row">
      <h3>Input event count per frame</h3>
      <span>Click a bar to inspect that frame</span>
    </div>

    <div class="encoding-bars" style={`grid-template-columns: repeat(${numSteps}, minmax(10px, 1fr));`}>
      {#each spikeCounts as count, i}
        <button type="button" class:active={Number(demoT) === i} on:click={() => jumpTo(i)} aria-label={`Jump to encoding timestep ${i}`}>
          <span style={`height:${Math.max(3, Math.round((count / Math.max(1, Math.max(...spikeCounts))) * 108))}px`}></span>
          {#if i === 0 || i === maxT || Number(demoT) === i}
            <em>{i}</em>
          {/if}
        </button>
      {/each}
    </div>
  </div>

  <div class="concept-strip">
    <article>
      <strong>Spike train</strong>
      <span>A time-indexed binary event stream.</span>
    </article>
    <article>
      <strong>Latency code</strong>
      <span>Information is carried by spike timing.</span>
    </article>
    <article>
      <strong>Reading rule</strong>
      <span>Early spikes usually represent stronger pixels.</span>
    </article>
  </div>
</section>
