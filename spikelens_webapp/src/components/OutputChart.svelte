<script>
  import * as d3 from 'd3';
  import { cumulativeOutputCounts } from '../lib/traceUtils.js';
  export let trace;
  export let t = 0;

  $: counts = cumulativeOutputCounts(trace, t);
  $: maxVal = Math.max(1, ...counts);
  const width = 330;
  const height = 230;
  const margin = {top: 12, right: 12, bottom: 32, left: 30};

  $: x = d3.scaleBand().domain(d3.range(10).map(String)).range([margin.left, width-margin.right]).padding(0.22);
  $: y = d3.scaleLinear().domain([0, maxVal]).nice().range([height-margin.bottom, margin.top]);
</script>

<svg class="output-chart" viewBox={`0 0 ${width} ${height}`}>
  {#each counts as value, i}
    <rect
      x={x(String(i))}
      y={y(value)}
      width={x.bandwidth()}
      height={y(0)-y(value)}
      class:pred={i === trace.pred_label}
      class:trueLabel={i === trace.true_label}
    />
    <text x={x(String(i)) + x.bandwidth()/2} y={height-12} text-anchor="middle">{i}</text>
    <text x={x(String(i)) + x.bandwidth()/2} y={y(value)-5} text-anchor="middle" class="bar-value">{value}</text>
  {/each}
  <text x={width/2} y={height-2} text-anchor="middle" class="axis-label">digit class</text>
</svg>
