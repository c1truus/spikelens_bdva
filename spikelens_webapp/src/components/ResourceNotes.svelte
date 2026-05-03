<section class="resource-notes card wide" aria-labelledby="resources-title">
  <div>
    <p class="eyebrow mini-eyebrow">References and implementation notes</p>
    <h2 id="resources-title">Resources used</h2>
    <p>
      SpikeLens is a static Svelte/Vite visualization over precomputed snnTorch traces. The project uses
      canvas heatmaps for dense layer grids and D3/SVG for timeline and output charts.
    </p>
  </div>

  <div class="resource-grid">
    <article>
      <h3>snnTorch spike encoding</h3>
      <p>
        Input conversion follows the latency-coding convention: stronger features fire earlier,
        weaker features fire later or remain silent.
      </p>
      <a href="https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html" target="_blank" rel="noreferrer">
        snnTorch Tutorial 1: Spike Encoding
      </a>
    </article>

    <article>
      <h3>LIF neuron model</h3>
      <p>
        The membrane-state interpretation follows snnTorch's Leaky Integrate-and-Fire neuron,
        where β controls membrane decay and threshold crossing produces spikes.
      </p>
      <a href="https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html" target="_blank" rel="noreferrer">
        snnTorch Leaky neuron documentation
      </a>
    </article>

    <article>
      <h3>Web visualization stack</h3>
      <p>
        The site is a client-side Svelte/Vite app. It loads static JSON traces from <code>public/data</code>
        and uses D3 for SVG path and scale utilities.
      </p>
      <a href="https://vite.dev/guide/assets" target="_blank" rel="noreferrer">Vite static assets</a>
      <a href="https://d3js.org/d3-shape/line" target="_blank" rel="noreferrer">D3 line generator</a>
    </article>
  </div>
</section>
