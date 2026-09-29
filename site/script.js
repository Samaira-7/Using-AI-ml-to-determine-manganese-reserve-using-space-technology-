// Live band-ratio calculator — mirrors spectral_indices.py exactly.
// Illustrative single-pixel math only; not a live satellite feed.
(function () {
  const EPS = 1e-6;
  const safeDivide = (n, d) => n / (d + EPS);

  const ids = ["blue", "red", "nir", "swir1", "swir2"];
  const sliders = {};
  ids.forEach((id) => {
    sliders[id] = document.getElementById("s-" + id);
  });

  // guard: calculator only exists on the landing page
  if (!sliders.blue) return;

  function readValues() {
    const v = {};
    ids.forEach((id) => {
      v[id] = parseFloat(sliders[id].value);
      document.getElementById("val-" + id).textContent = v[id].toFixed(3);
    });
    return v;
  }

  function ironOxideIndex(red, blue) {
    return safeDivide(red, blue);
  }
  function ferrousIronIndex(swir1, nir) {
    return safeDivide(swir1, nir);
  }
  function clayAlterationIndex(swir1, swir2) {
    return safeDivide(swir1, swir2);
  }
  function ndvi(nir, red) {
    return safeDivide(nir - red, nir + red);
  }

  function setBar(barId, value, min, max) {
    const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
    document.getElementById(barId).style.width = pct + "%";
  }

  function update() {
    const v = readValues();

    const mori = ironOxideIndex(v.red, v.blue);
    const hai = clayAlterationIndex(v.swir1, v.swir2);
    const fmi = ferrousIronIndex(v.swir1, v.nir);
    const veg = ndvi(v.nir, v.red);

    document.getElementById("out-mori").textContent = mori.toFixed(2);
    document.getElementById("out-hai").textContent = hai.toFixed(2);
    document.getElementById("out-fmi").textContent = fmi.toFixed(2);
    document.getElementById("out-ndvi").textContent = veg.toFixed(2);

    setBar("bar-mori", mori, 0, 3);
    setBar("bar-hai", hai, 0, 3);
    setBar("bar-fmi", fmi, 0, 2);
    setBar("bar-ndvi", veg, -0.2, 0.8);

    // Composite score, matching manganese_composite_index() logic loosely
    // (single-pixel illustration — the real function normalises across
    // a whole raster, which a single slider value can't reproduce).
    const composite = 0.5 * Math.min(mori / 2.5, 1) + 0.3 * Math.min(hai / 2.5, 1) - 0.2 * Math.max(veg, 0);

    const verdictEl = document.getElementById("calc-verdict");
    if (veg > 0.3) {
      verdictEl.innerHTML =
        "NDVI is high enough that this pixel would likely be <strong>masked as vegetated</strong> — surface signal underneath is not visible to the sensor.";
    } else if (composite > 0.55) {
      verdictEl.innerHTML =
        "Composite score sits in the <strong>high</strong> screening range — a candidate for field follow-up, not a confirmed deposit.";
    } else if (composite > 0.3) {
      verdictEl.innerHTML =
        "Composite score sits in the <strong>moderate</strong> screening range — worth a look, not a headline.";
    } else {
      verdictEl.innerHTML =
        "Composite score sits in the <strong>low</strong> screening range for this illustrative pixel.";
    }
  }

  ids.forEach((id) => sliders[id].addEventListener("input", update));
  update();
})();
