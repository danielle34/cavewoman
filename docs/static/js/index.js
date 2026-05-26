/* CAVEWOMAN. Small page JS.
 * - L0..L4 level demo tab switcher (Bulma is-toggle-rounded tabs)
 * Template scaffold adapted from Nerfies (CC BY-SA 4.0). */

(function () {
  "use strict";

  function initLevelDemo() {
    var tabs = document.querySelectorAll(".level-tabs a[data-level]");
    var panels = document.querySelectorAll(".demo-panel");
    if (!tabs.length) return;

    function show(level) {
      tabs.forEach(function (a) {
        var on = a.getAttribute("data-level") === level;
        a.parentNode.classList.toggle("is-active", on);
      });
      panels.forEach(function (p) {
        p.classList.toggle("is-active",
          p.getAttribute("data-level") === level);
      });
    }

    tabs.forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        show(a.getAttribute("data-level"));
      });
    });
  }

  document.addEventListener("DOMContentLoaded", initLevelDemo);
})();
