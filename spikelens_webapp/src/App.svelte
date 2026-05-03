<script>
  import { onMount, onDestroy } from 'svelte';

  import NetworkScene from './components/NetworkScene.svelte';
  import OutputChart from './components/OutputChart.svelte';
  import TimelineChart from './components/TimelineChart.svelte';
  import DigitPreview from './components/DigitPreview.svelte';
  import LayerSummary from './components/LayerSummary.svelte';
  import LatencyEncodingExplainer from './components/LatencyEncodingExplainer.svelte';
  import HyperparameterExplainer from './components/HyperparameterExplainer.svelte';
  import ResourceNotes from './components/ResourceNotes.svelte';

  let manifest = null;
  let trace = null;

  let selectedBeta = 'all';
  let selectedLatency = 'all';
  let selectedSteps = 'all';
  let selectedSampleIndex = 0;

  let digitFilter = 'all';
  let correctnessFilter = 'all';

  let t = 0;
  let mode = 'membrane';
  let playing = false;
  let fps = 1;
  let timer = null;
  let errorMessage = '';
  let loadingTrace = false;

  $: allVariants = manifest?.variants ?? [];

  function nstr(x) {
    if (x === undefined || x === null) return '';
    return String(Number(x));
  }

  function sortNumberStrings(arr) {
    return [...new Set(arr.filter((x) => x !== undefined && x !== null).map((x) => String(Number(x))))]
      .sort((a, b) => Number(a) - Number(b));
  }

  $: betaOptions = sortNumberStrings(allVariants.map((v) => v.beta));
  $: latencyOptions = sortNumberStrings(allVariants.map((v) => v.latency_threshold));
  $: stepOptions = sortNumberStrings(allVariants.map((v) => v.num_steps));

  $: matchingVariants = allVariants.filter((v) => {
    if (selectedBeta !== 'all' && nstr(v.beta) !== selectedBeta) return false;
    if (selectedLatency !== 'all' && nstr(v.latency_threshold) !== selectedLatency) return false;
    if (selectedSteps !== 'all' && nstr(v.num_steps) !== selectedSteps) return false;
    return true;
  });

  // With full beta/latency/steps selected this should be one model.
  // If a selector is set to all, use the first matching model.
  $: selectedVariant = matchingVariants?.[0];
  $: rawSamples = selectedVariant?.samples ?? [];

  $: filteredSamples = rawSamples.filter((s) => {
    if (digitFilter !== 'all' && Number(s.true_label) !== Number(digitFilter)) return false;
    if (correctnessFilter === 'correct' && s.correct !== true) return false;
    if (correctnessFilter === 'wrong' && s.correct !== false) return false;
    return true;
  });

  $: selectedSample = filteredSamples?.[Number(selectedSampleIndex)];
  $: maxT = trace ? Number(trace.num_steps ?? selectedVariant?.num_steps ?? 25) - 1 : 24;
  $: totalModelCount = allVariants.length;
  $: totalTraceCount = allVariants.reduce((acc, v) => acc + Number(v.sample_count ?? v.samples?.length ?? 0), 0);

  function dataUrl(path) {
    const base = import.meta.env.BASE_URL || '/';
    const cleanBase = base.endsWith('/') ? base : `${base}/`;
    const cleanPath = String(path).replace(/^\/+/, '');
    return `${cleanBase}${cleanPath}`;
  }

  async function fetchJson(path) {
    const url = dataUrl(path);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
    return await res.json();
  }

  function normalizeManifest(raw) {
    const m = structuredClone(raw);
    if (!Array.isArray(m.variants)) throw new Error('manifest.json must contain a variants array.');

    m.variants = m.variants
      .map((v) => ({
        ...v,
        beta: Number(v.beta),
        latency_threshold: Number(v.latency_threshold),
        num_steps: Number(v.num_steps),
        samples: Array.isArray(v.samples) ? v.samples : [],
        sample_count: v.sample_count ?? v.samples?.length ?? 0,
        correct_count: v.correct_count ?? v.samples?.filter((s) => s.correct === true).length ?? 0,
        wrong_count: v.wrong_count ?? v.samples?.filter((s) => s.correct === false).length ?? 0
      }))
      .sort((a, b) =>
        Number(a.beta) - Number(b.beta) ||
        Number(a.latency_threshold) - Number(b.latency_threshold) ||
        Number(a.num_steps) - Number(b.num_steps)
      );

    return m;
  }

  function normalizeTrace(rawTrace) {
    const out = structuredClone(rawTrace);

    if (!out.num_steps) out.num_steps = out.timesteps ?? selectedVariant?.num_steps ?? 25;

    if (!out.architecture) {
      out.architecture = manifest?.architecture ?? {
        input: 784,
        fc1: 1024,
        fc2: 256,
        out: 10,
        grid_shapes: { input: [28, 28], fc1: [32, 32], fc2: [16, 16], out: [1, 10] }
      };
    }

    if (!out.spikes_dense && out.spikes) out.spikes_dense = out.spikes;
    if (!out.spikes_dense && out.events) out.spikes_dense = buildDenseSpikesFromEvents(out);
    if (!out.membrane && out.mem) out.membrane = out.mem;
    if (!out.output_spike_counts && out.spikes_dense?.out) out.output_spike_counts = cumulativeOutputCounts(out.spikes_dense.out);

    out.beta = out.beta ?? selectedVariant?.beta;
    out.latency_threshold = out.latency_threshold ?? out.threshold ?? selectedVariant?.latency_threshold;
    out.model_id = out.model_id ?? selectedVariant?.model_id;

    return out;
  }

  function buildDenseSpikesFromEvents(traceObj) {
    const steps = Number(traceObj.num_steps ?? traceObj.timesteps ?? selectedVariant?.num_steps ?? 25);
    const sizes = {
      input: traceObj.architecture?.input ?? 784,
      fc1: traceObj.architecture?.fc1 ?? 1024,
      fc2: traceObj.architecture?.fc2 ?? 256,
      out: traceObj.architecture?.out ?? 10
    };

    const dense = {};
    for (const layer of Object.keys(sizes)) {
      dense[layer] = Array.from({ length: steps }, () => Array.from({ length: sizes[layer] }, () => 0));
      const evs = traceObj.events?.[layer] ?? [];
      for (const e of evs) {
        const step = Number(e.t);
        const neuronId = Number(e.neuron_id ?? e.n);
        if (Number.isFinite(step) && Number.isFinite(neuronId) && step >= 0 && step < steps && neuronId >= 0 && neuronId < sizes[layer]) {
          dense[layer][step][neuronId] = 1;
        }
      }
    }
    return dense;
  }

  function cumulativeOutputCounts(outSpikes) {
    const counts = Array(10).fill(0);
    for (const frame of outSpikes) {
      for (let i = 0; i < Math.min(10, frame.length); i++) counts[i] += Number(frame[i] ?? 0);
    }
    return counts;
  }

  async function loadManifest() {
    try {
      errorMessage = '';
      trace = null;

      const raw = await fetchJson('data/manifest.json');
      const nextManifest = normalizeManifest(raw);
      if (!nextManifest.variants || nextManifest.variants.length === 0) {
        throw new Error('manifest.json loaded, but it contains 0 model variants. Re-run export, then rebuild manifest.');
      }

      manifest = nextManifest;

      const firstVariant = nextManifest.variants[0];
      selectedBeta = nstr(firstVariant.beta);
      selectedLatency = nstr(firstVariant.latency_threshold);
      selectedSteps = nstr(firstVariant.num_steps);
      selectedSampleIndex = 0;

      const firstSample = firstVariant.samples?.[0];
      if (!firstSample) throw new Error(`First model variant "${firstVariant.model_id}" has 0 samples.`);

      loadingTrace = true;
      const rawTrace = await fetchJson(firstSample.path);
      trace = normalizeTrace(rawTrace);
      t = 0;
    } catch (err) {
      console.error('[SpikeLens] loadManifest failed:', err);
      errorMessage = err?.message ?? String(err);
    } finally {
      loadingTrace = false;
    }
  }

  async function loadTrace() {
    try {
      errorMessage = '';
      if (!selectedVariant) {
        trace = null;
        return;
      }
      if (!selectedSample) {
        trace = null;
        return;
      }

      loadingTrace = true;
      const rawTrace = await fetchJson(selectedSample.path);
      trace = normalizeTrace(rawTrace);
      t = 0;
    } catch (err) {
      console.error('[SpikeLens] loadTrace failed:', err);
      errorMessage = err?.message ?? String(err);
    } finally {
      loadingTrace = false;
    }
  }

  function resetSampleAndLoad() {
    selectedSampleIndex = 0;
    t = 0;
    setTimeout(loadTrace, 0);
  }

  function togglePlay() {
    playing = !playing;
  }

  function stopTimer() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  $: {
    stopTimer();
    if (playing && trace) {
      timer = setInterval(() => {
        t = Number(t) >= maxT ? 0 : Number(t) + 1;
      }, 1000 / Number(fps));
    }
  }

  $: if (manifest && selectedSampleIndex >= filteredSamples.length && filteredSamples.length > 0) {
    selectedSampleIndex = 0;
  }

  onMount(loadManifest);
  onDestroy(stopTimer);
</script>

<svelte:head>
  <title>SpikeLens | Interactive SNN Inference Visualization</title>
</svelte:head>


<main class="app app-polished">
  <header class="hero hero-polished">
    <div>
      <p class="eyebrow">Big Data Visualization and Visual Analysis Final Project</p>
      <h1>SpikeLens</h1>
      <p class="subtitle">
        Explore precomputed SNN inference traces across membrane decay β, latency threshold,
        inference timesteps, input samples, spike events, and membrane potentials.
      </p>
    </div>

    {#if trace}
      <div class="status-card">
        <div class="status-title">Current trace</div>
        <div class="prediction">
          <span>true={trace.true_label}</span>
          <span>pred={trace.pred_label}</span>
          <span class:ok={trace.correct} class:bad={!trace.correct}>{trace.correct ? 'correct' : 'wrong'}</span>
        </div>
        <div class="tiny">
          model={trace.model_id ?? selectedVariant?.model_id}<br />
          β={trace.beta ?? selectedVariant?.beta ?? '—'} · lat={trace.latency_threshold ?? selectedVariant?.latency_threshold ?? '—'} · t={t}/{maxT}
        </div>
      </div>
    {/if}
  </header>

  {#if errorMessage}
    <section class="card wide error-box">
      <h2>SpikeLens failed to load data</h2>
      <p>{errorMessage}</p>
    </section>
  {:else if !manifest}
    <div class="loading">Loading SpikeLens manifest…</div>
  {:else}
    <section class="dataset-panel card wide compact-card">
      <div>
        <h2>Dataset browser</h2>
        <p class="tiny">Loaded {totalModelCount} model variants and {totalTraceCount} trace files from manifest.</p>
      </div>
      <div class="dataset-stats">
        <span>{matchingVariants.length} matching model{matchingVariants.length === 1 ? '' : 's'}</span>
        <span>{rawSamples.length} samples in selected model</span>
        <span>{filteredSamples.length} samples after filter</span>
      </div>
    </section>

    <section class="controls-panel controls-panel-v2">
      <div class="control-group small-control">
        <label for="beta-select">β membrane decay</label>
        <select id="beta-select" bind:value={selectedBeta} on:change={resetSampleAndLoad}>
          <option value="all">all β</option>
          {#each betaOptions as b}<option value={b}>β = {b}</option>{/each}
        </select>
      </div>

      <div class="control-group small-control">
        <label for="lat-select">Latency threshold</label>
        <select id="lat-select" bind:value={selectedLatency} on:change={resetSampleAndLoad}>
          <option value="all">all latencies</option>
          {#each latencyOptions as l}<option value={l}>lat = {l}</option>{/each}
        </select>
      </div>

      <div class="control-group small-control">
        <label for="steps-select">Inference steps</label>
        <select id="steps-select" bind:value={selectedSteps} on:change={resetSampleAndLoad}>
          <option value="all">all steps</option>
          {#each stepOptions as s}<option value={s}>{s} steps</option>{/each}
        </select>
      </div>

      <div class="control-group small-control">
        <label for="digit-filter">Digit</label>
        <select id="digit-filter" bind:value={digitFilter} on:change={resetSampleAndLoad}>
          <option value="all">all digits</option>
          {#each Array(10) as _, i}<option value={i}>label {i}</option>{/each}
        </select>
      </div>

      <div class="control-group small-control">
        <label for="correctness-filter">Result</label>
        <select id="correctness-filter" bind:value={correctnessFilter} on:change={resetSampleAndLoad}>
          <option value="all">all</option>
          <option value="correct">correct only</option>
          <option value="wrong">wrong only</option>
        </select>
      </div>

      <div class="control-group sample-control">
        <label for="sample-select">Input sample</label>
        <select id="sample-select" bind:value={selectedSampleIndex} on:change={loadTrace}>
          {#each filteredSamples as s, i}
            <option value={i}>sample {s.sample_idx ?? i} · true={s.true_label} · pred={s.pred_label} {s.correct ? '✓' : '✗'}</option>
          {/each}
        </select>
      </div>

      <div class="control-group mode-control">
        <label>Layer coloring</label>
        <div class="segmented" role="group" aria-label="Layer coloring mode">
          <button type="button" class:active={mode === 'spike'} on:click={() => mode = 'spike'}>spike</button>
          <button type="button" class:active={mode === 'membrane'} on:click={() => mode = 'membrane'}>membrane</button>
          <button type="button" class:active={mode === 'cumulative'} on:click={() => mode = 'cumulative'}>cumulative</button>
        </div>
      </div>

      <div class="control-group timeline-control timeline-control-v2">
        <label for="time-slider">Timestep t={t}</label>
        <input id="time-slider" type="range" min="0" max={maxT} bind:value={t} />
        <div class="play-row">
          <button type="button" class="primary" on:click={togglePlay} disabled={!trace}>{playing ? 'Pause' : 'Play'}</button>
          <select aria-label="Playback speed" bind:value={fps}>
            <option value={1}>1 step/s</option>
            <option value={2}>2 step/s</option>
            <option value={4}>4 step/s</option>
            <option value={12}>12 step/s</option>
            <option value={24}>24 step/s</option>
          </select>
        </div>
      </div>
    </section>

    {#if loadingTrace}
      <div class="loading">Loading selected trace…</div>
    {:else if !trace}
      <section class="card wide"><h2>No trace selected</h2><p class="explain">Try changing the β, latency, steps, digit, or correctness filters.</p></section>
    {:else}
      <section class="layout layout-v2">
        <aside class="left-rail card">
          <h2>Input</h2>
          <DigitPreview image={trace.original_image} spikeFrame={trace.spikes_dense?.input?.[Number(t)]} />
          <p class="explain">The original MNIST digit is latency encoded, then replayed as a short spike sequence.</p>
          <LayerSummary {trace} {t} />
        </aside>

        <section class="center-stage card">
          <div class="stage-header stage-header-v2">
            <div>
              <h2>Network trace</h2>
              <p>28×28 input → 32×32 hidden layer → 16×16 hidden layer → 10 output classes</p>
            </div>
            <div class="variant-pill">β={selectedVariant?.beta} · lat={selectedVariant?.latency_threshold} · steps={selectedVariant?.num_steps}</div>
          </div>
          <NetworkScene {trace} {t} {mode} />
        </section>

        <aside class="right-rail card">
          <h2>Output decision</h2>
          <OutputChart {trace} {t} />
          <p class="explain">Bars show cumulative output spikes up to the current timestep.</p>
        </aside>
      </section>

      <section class="card wide"><h2>Layer spike timeline</h2><TimelineChart {trace} bind:t /></section>

      <LatencyEncodingExplainer {trace} />
      <HyperparameterExplainer
  beta={selectedVariant?.beta}
  latencyThreshold={selectedVariant?.latency_threshold}
  numSteps={trace?.num_steps ?? selectedVariant?.num_steps}
/>
    {/if}

    <section class="report card wide concept-brief">
      <div class="brief-heading">
        <p class="eyebrow mini-eyebrow">Reading guide</p>
        <h2>How to read SpikeLens</h2>
      </div>

      <div class="brief-grid">
        <article>
          <span class="brief-term">Spike event</span>
          <p>A binary signal at one timestep. In spike mode, a lit cell means that neuron fired now.</p>
        </article>

        <article>
          <span class="brief-term">Membrane state</span>
          <p>The neuron's accumulated evidence. In membrane mode, black is low and white is high.</p>
        </article>

        <article>
          <span class="brief-term">Output evidence</span>
          <p>The class bars count output spikes over time. The predicted digit is the largest count.</p>
        </article>

        <article>
          <span class="brief-term">Comparison workflow</span>
          <p>Keep the same sample, then change β, latency threshold, or steps to compare propagation.</p>
        </article>
      </div>
    </section>

    <ResourceNotes />
  {/if}
</main>