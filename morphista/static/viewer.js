// Tiled slide viewer (OpenSeadragon). Wheel zooms at the cursor, drag pans, double-click zooms in;
// 100 % = one image pixel per screen pixel. The zoom bar is wired here because NiceGUI strips onclick.
(() => {
  let viewer = null;
  const label = () => {
    const el = document.querySelector('.mh-zoombar .lvl'); if (!el || !viewer || !viewer.world.getItemAt(0)) return;
    const pct = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true)) * 100;
    el.textContent = (pct >= 10 ? Math.round(pct) : pct.toFixed(1)) + ' %';
  };
  window.mhShow = (key, w, h) => {
    if (!viewer) {
      viewer = OpenSeadragon({
        element: document.getElementById('mh-osd'), showNavigationControl: false,
        showNavigator: true, navigatorPosition: 'TOP_RIGHT', navigatorSizeRatio: 0.18,
        maxZoomPixelRatio: 4, visibilityRatio: 0.3, animationTime: 0.35, springStiffness: 9,
        gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true},
      });
      viewer.addHandler('animation', label); viewer.addHandler('open', label); viewer.addHandler('resize', label);
    }
    viewer.open({height: h, width: w, tileSize: 512, tileOverlap: 0,
                 getTileUrl: (l, x, y) => `/tiles/${key}/${l}/${x}_${y}.jpg`});
  };
  document.addEventListener('click', e => {
    const b = e.target.closest && e.target.closest('.mh-zoombar button'); if (!b || !viewer) return;
    const vp = viewer.viewport, act = b.dataset.act;
    if (act === 'in') vp.zoomBy(1.6); else if (act === 'out') vp.zoomBy(1 / 1.6);
    else if (act === 'one') vp.zoomTo(vp.imageToViewportZoom(1)); else vp.goHome();
    vp.applyConstraints();
  });
})();
