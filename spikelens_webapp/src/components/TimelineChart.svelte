<script>
  import * as d3 from 'd3';
  import { layerSpikeSeries } from '../lib/traceUtils.js';

  export let trace;
  export let t = 0;

  const width = 920;
  const height = 240;
  const margin = {top: 20, right: 20, bottom: 32, left: 48};
  const layers = ['input', 'fc1', 'fc2', 'out'];

  $: series = layers.map(layer => ({
    layer,
    values: layerSpikeSeries(trace, layer)
  }));

  $: maxT = trace.num_steps - 1;
  $: maxY = Math.max(1, ...series.flatMap(s => s.values));
  $: x = d3.scaleLinear().domain([0, maxT]).range([margin.left, width-margin.right]);
  $: y = d3.scaleLinear().domain([0, maxY]).nice().range([height-margin.bottom, margin.top]);
  $: color = d3.scaleOrdinal().domain(layers).range(['#38bdf8','#a78bfa','#f59e0b','#22c55e']);
  $: line = d3.line().x((d,i)=>x(i)).y(d=>y(d));

  function clickTimeline(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const localX = px / rect.width * width;
    const inv = Math.round(x.invert(localX));
    t = Math.max(0, Math.min(maxT, inv));
  }
</script>

<svg class="timeline-chart" viewBox={`0 0 ${width} ${height}`} on:click={clickTimeline}>
  {#each series as s}
    <path d={line(s.values)} fill="none" stroke={color(s.layer)} stroke-width="3" opacity="0.86" />
  {/each}

  <line x1={x(t)} x2={x(t)} y1={margin.top} y2={height-margin.bottom} stroke="#f8fafc" stroke-width="2" stroke-dasharray="5 4" />

  {#each series as s, idx}
    <circle cx={width-130} cy={30+idx*22} r="6" fill={color(s.layer)} />
    <text x={width-115} y={35+idx*22}>{s.layer}</text>
  {/each}

  <text x={width/2} y={height-5} text-anchor="middle" class="axis-label">click timeline to jump to timestep</text>
</svg>
