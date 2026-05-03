<script>
  export let beta = 0.65;
  export let latencyThreshold = 0.15;
  export let numSteps = 25;

  function fmt(value, fallback = '—') {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return fallback;
    return Number(value).toFixed(2).replace(/\.00$/, '');
  }

  $: betaValue = Number(beta ?? 0.65);
  $: latencyValue = Number(latencyThreshold ?? 0.15);
  $: stepsValue = Number(numSteps ?? 25);

  $: betaClamped = Math.max(0, Math.min(1, betaValue));
  $: betaMemoryPct = Math.round(betaClamped * 100);

  $: latencyMin = 0.15;
  $: latencyMax = 0.35;
  $: latencyPct = Math.round(
    100 * Math.max(0, Math.min(1, (latencyValue - latencyMin) / (latencyMax - latencyMin || 1)))
  );

  $: stepsMin = 25;
  $: stepsMax = 60;
  $: stepsPct = Math.round(
    100 * Math.max(0, Math.min(1, (stepsValue - stepsMin) / (stepsMax - stepsMin || 1)))
  );

  $: decayCells = Array.from({ length: 10 }, (_, i) => Math.pow(betaClamped, i));
  $: frames = Array.from({ length: 12 }, (_, i) => i / 11 <= stepsPct / 100);
</script>

<section class="hyper-explainer card wide editorial-section" aria-labelledby="hyper-title">
  <div class="section-kicker">Model controls</div>

  <div class="hyper-hero editorial-head">
    <div>
      <h2 id="hyper-title">What the three controls mean</h2>
      <p class="hyper-lead editorial-lead">
        These parameters control memory, input sparsity, and inference duration. Keep the same digit selected,
        change one parameter, then compare how activity moves through the layers.
      </p>
    </div>

    <div class="current-hyper-card compact-current-card">
      <span>Current run</span>
      <strong>β={fmt(betaValue)} · latency={fmt(latencyValue)} · steps={stepsValue}</strong>
    </div>
  </div>

  <div class="hyper-grid editorial-hyper-grid">
    <article class="hyper-card beta-card compact-hyper-card">
      <div class="hyper-number">β</div>
      <div class="hyper-copy">
        <h3>Membrane memory</h3>
        <p>
          β is the decay factor of the LIF membrane. Higher β keeps past input longer;
          lower β makes neurons depend more on recent spikes.
        </p>

        <div class="big-equation">membrane_next ≈ β · membrane_old + input</div>

        <div class="meter-block">
          <div class="meter-label">
            <span>short memory</span>
            <strong>{betaMemoryPct}% retained</strong>
            <span>long memory</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill" style={`width:${betaMemoryPct}%`}></div>
          </div>
        </div>

        <div class="decay-demo" aria-label="membrane decay sketch">
          {#each decayCells as v, i}
            <span class="decay-cell" style={`opacity:${0.14 + 0.86 * v}; transform:scale(${0.72 + 0.28 * v})`} title={`t+${i}: ${(v * 100).toFixed(1)}% remains`}></span>
          {/each}
        </div>

        <p class="human-note"><strong>Interpretation:</strong> larger β usually supports slower evidence accumulation.</p>
      </div>
    </article>

    <article class="hyper-card latency-card compact-hyper-card">
      <div class="hyper-number">lat</div>
      <div class="hyper-copy">
        <h3>Input cutoff</h3>
        <p>
          The latency threshold determines which pixels are strong enough to become spike events.
          Raising it makes the input stream sparser.
        </p>

        <div class="threshold-demo compact-threshold-demo">
          {#each Array.from({ length: 12 }) as _, i}
            {@const brightness = 1 - i / 11}
            {@const active = brightness >= latencyValue}
            <div class="threshold-column">
              <div class:inactive={!active} class="brightness-bar" style={`height:${Math.round(18 + brightness * 72)}px; opacity:${0.25 + brightness * 0.75}`}></div>
              <small>{active ? 'spk' : '—'}</small>
            </div>
          {/each}
        </div>

        <div class="meter-block">
          <div class="meter-label">
            <span>dense input</span>
            <strong>threshold {fmt(latencyValue)}</strong>
            <span>sparse input</span>
          </div>
          <div class="meter-track latency-track">
            <div class="meter-fill latency-fill" style={`width:${latencyPct}%`}></div>
          </div>
        </div>

        <p class="human-note"><strong>Interpretation:</strong> higher threshold removes weaker pixel events.</p>
      </div>
    </article>

    <article class="hyper-card steps-card compact-hyper-card">
      <div class="hyper-number">T</div>
      <div class="hyper-copy">
        <h3>Evidence window</h3>
        <p>
          Steps are the number of frames in the inference movie. More steps give late spikes more time
          to affect the output, but inference takes longer.
        </p>

        <div class="frames-demo compact-frames-demo" aria-label="time frame sketch">
          {#each frames as active, i}
            <span class:active-frame={active}>{i === 0 ? '0' : i === 11 ? 'T' : ''}</span>
          {/each}
        </div>

        <div class="meter-block">
          <div class="meter-label">
            <span>fast</span>
            <strong>{stepsValue} frames</strong>
            <span>longer context</span>
          </div>
          <div class="meter-track steps-track">
            <div class="meter-fill steps-fill" style={`width:${stepsPct}%`}></div>
          </div>
        </div>

        <p class="human-note"><strong>Interpretation:</strong> more steps can improve temporal evidence, at higher cost.</p>
      </div>
    </article>
  </div>

  <div class="hyper-howto compact-howto">
    <div>
      <h3>Recommended comparison</h3>
      <p>Hold the digit fixed. Change one parameter at a time, then inspect the hidden-layer density, output spike count, and timeline peaks.</p>
    </div>
    <div>
      <h3>Visual convention</h3>
      <p>Spike mode shows binary events. Membrane mode shows state intensity. Cumulative mode shows activity accumulated so far.</p>
    </div>
  </div>
</section>
