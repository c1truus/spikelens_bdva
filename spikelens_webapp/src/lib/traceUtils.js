export function layerShape(trace, layer) {
  const s = trace?.architecture?.grid_shapes?.[layer];
  if (!s) return [1, 1];
  return s;
}

export function arrayMinMax(arr) {
  let min = Infinity;
  let max = -Infinity;
  const walk = (x) => {
    if (Array.isArray(x)) {
      for (const v of x) walk(v);
    } else if (Number.isFinite(x)) {
      if (x < min) min = x;
      if (x > max) max = x;
    }
  };
  walk(arr);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

export function getLayerValues(trace, layer, t, mode) {
  if (!trace) return [];
  if (mode === 'membrane' && trace.membrane_dense?.[layer]) {
    return trace.membrane_dense[layer][t] ?? [];
  }
  if (mode === 'cumulative') {
    const dense = trace.spikes_dense?.[layer] ?? [];
    const n = dense?.[0]?.length ?? 0;
    const out = new Array(n).fill(0);
    for (let i = 0; i <= t; i++) {
      const row = dense[i] ?? [];
      for (let j = 0; j < n; j++) out[j] += row[j] ? 1 : 0;
    }
    return out;
  }
  return trace.spikes_dense?.[layer]?.[t] ?? [];
}

export function getSpikeMask(trace, layer, t) {
  return trace?.spikes_dense?.[layer]?.[t] ?? [];
}

export function cumulativeOutputCounts(trace, t) {
  const dense = trace?.spikes_dense?.out ?? [];
  const out = new Array(10).fill(0);
  for (let i = 0; i <= t; i++) {
    const row = dense[i] ?? [];
    for (let j = 0; j < 10; j++) out[j] += row[j] ? 1 : 0;
  }
  return out;
}

export function layerSpikeSeries(trace, layer) {
  if (trace?.layer_spikes_per_timestep?.[layer]) return trace.layer_spikes_per_timestep[layer];
  const dense = trace?.spikes_dense?.[layer] ?? [];
  return dense.map(row => row.reduce((a,b)=>a+(b?1:0),0));
}
