<script>
  import { onMount, afterUpdate } from 'svelte';
  export let image = [];
  export let spikeFrame = [];

  let canvas;
  const size = 168;
  const grid = 28;

  function draw() {
    if (!canvas || !image?.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cell = size / grid;
    ctx.clearRect(0,0,size,size);

    for (let r=0; r<grid; r++) {
      for (let c=0; c<grid; c++) {
        const idx = r*grid+c;
        const v = image?.[r]?.[c] ?? 0;
        const spk = spikeFrame?.[idx] ?? 0;
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(c*cell, r*cell, cell, cell);
        if (spk) {
          ctx.fillStyle = 'rgba(255, 200, 0, 0.95)';
          ctx.fillRect(c*cell, r*cell, cell, cell);
        }
      }
    }
  }

  onMount(draw);
  afterUpdate(draw);
</script>

<canvas bind:this={canvas} class="digit-canvas"></canvas>
<div class="hint">yellow = input pixel spiked at current timestep</div>
