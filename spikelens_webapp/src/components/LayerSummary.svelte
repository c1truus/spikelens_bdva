<script>
  import { layerSpikeSeries } from '../lib/traceUtils.js';
  export let trace;
  export let t = 0;
  const layers = ['input', 'fc1', 'fc2', 'out'];

  function total(layer) {
    return layerSpikeSeries(trace, layer).reduce((a,b)=>a+b,0);
  }
  function now(layer) {
    return layerSpikeSeries(trace, layer)[t] ?? 0;
  }
</script>

<div class="summary">
  <h3>Layer spikes</h3>
  {#each layers as layer}
    <div class="summary-row">
      <span>{layer}</span>
      <b>{now(layer)}</b>
      <em>total {total(layer)}</em>
    </div>
  {/each}
</div>
