/*
 * why_price.js - "Why this price?" section for LandInsight (quick + advanced).
 *
 * Usage:
 *   LandWhy.render(document.getElementById("whyPrice"), explanation);
 *
 * `explanation` is the JSON produced by explainer.py. No chart library is
 * needed: the waterfall is drawn with plain HTML/CSS so it works offline.
 */
(function () {
    "use strict";

    var CSS = "\
.why-card{margin-top:26px;padding:24px;border-radius:18px;background:#fff;border:1px solid #e5e7eb;box-shadow:0 10px 30px rgba(15,23,42,.06);font-family:inherit;text-align:left}\
.why-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px}\
.why-head h3{font-size:20px;color:#111827;margin:0}\
.why-toggle{display:inline-flex;background:#f1f5f9;border-radius:10px;padding:3px}\
.why-toggle button{border:0;background:transparent;padding:7px 13px;font-size:13px;font-weight:700;color:#64748b;border-radius:8px;cursor:pointer}\
.why-toggle button.on{background:#fff;color:#1d4ed8;box-shadow:0 1px 4px rgba(15,23,42,.12)}\
.why-sub{color:#64748b;font-size:13px;margin-bottom:14px}\
.why-summary{background:#f8fafc;border-left:4px solid #2563eb;border-radius:8px;padding:12px 14px;font-size:14px;line-height:1.6;color:#334155;margin-bottom:18px}\
.why-row{display:grid;grid-template-columns:minmax(120px,210px) 1fr 92px;align-items:center;gap:10px;padding:5px 0}\
.why-name{font-size:13px;color:#334155;font-weight:600;line-height:1.25}\
.why-name small{display:block;color:#94a3b8;font-weight:500;font-size:11px}\
.why-track{position:relative;height:20px;background:#f1f5f9;border-radius:6px}\
.why-bar{position:absolute;top:0;height:100%;border-radius:5px;min-width:3px}\
.why-bar.up{background:#16a34a}.why-bar.down{background:#dc2626}.why-bar.base{background:#94a3b8}.why-bar.final{background:#2563eb}\
.why-val{font-size:13px;font-weight:800;text-align:right;white-space:nowrap}\
.why-val.up{color:#15803d}.why-val.down{color:#b91c1c}.why-val.flat{color:#334155}\
.why-tag{display:inline-block;margin-left:5px;padding:1px 6px;border-radius:6px;background:#fef3c7;color:#92400e;font-size:10px;font-weight:700}\
.why-note{margin-top:16px;color:#94a3b8;font-size:12px;line-height:1.5}\
.why-skel{height:18px;border-radius:6px;margin:10px 0;background:linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 37%,#f1f5f9 63%);background-size:400% 100%;animation:whyShimmer 1.3s infinite}\
@keyframes whyShimmer{0%{background-position:100% 50%}100%{background-position:0 50%}}\
@media(max-width:640px){.why-row{grid-template-columns:1fr 84px}.why-track{grid-column:1/-1;order:3}}";

    function injectCss() {
        if (document.getElementById("why-price-css")) { return; }
        var style = document.createElement("style");
        style.id = "why-price-css";
        style.textContent = CSS;
        document.head.appendChild(style);
    }

    function esc(value) {
        return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
            return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
        });
    }

    function inr(value, signed) {
        var n = Math.round(Math.abs(Number(value) || 0));
        var text = "\u20B9" + n.toLocaleString("en-IN");
        if (signed) { return (Number(value) >= 0 ? "+" : "-") + text; }
        return (Number(value) < 0 ? "-" : "") + text;
    }

    function row(name, sub, tag, left, width, cls, valueText, valueCls) {
        return '<div class="why-row">' +
            '<div class="why-name">' + esc(name) + (tag ? '<span class="why-tag">assumed</span>' : "") +
            (sub ? "<small>" + esc(sub) + "</small>" : "") + "</div>" +
            '<div class="why-track"><div class="why-bar ' + cls + '" style="left:' + left.toFixed(2) +
            "%;width:" + width.toFixed(2) + '%"></div></div>' +
            '<div class="why-val ' + valueCls + '">' + valueText + "</div></div>";
    }

    function waterfallHtml(ex) {
        var steps = ex.waterfall || [];
        var run = ex.base_value;
        var points = [{lo: 0, hi: ex.base_value}];
        steps.forEach(function (s) {
            var next = run + s.impact;
            points.push({lo: Math.min(run, next), hi: Math.max(run, next)});
            run = next;
        });
        points.push({lo: 0, hi: ex.predicted_price});

        var max = 0;
        points.forEach(function (p) { max = Math.max(max, p.hi); });
        // zoom: start the axis a little below the smallest value so small steps stay visible
        var min = Infinity;
        points.forEach(function (p) { if (p.lo > 0) { min = Math.min(min, p.lo); } });
        points[0].lo = points[points.length - 1].lo = 0;
        var floor = Math.max(0, Math.min(min === Infinity ? 0 : min, ex.base_value, ex.predicted_price) * 0.8);
        var span = (max - floor) || 1;
        function pos(a, b) {
            var left = ((Math.max(a, floor) - floor) / span) * 100;
            var width = ((b - Math.max(a, floor)) / span) * 100;
            return [left, width];
        }

        var html = "";
        var p0 = pos(floor, ex.base_value);
        html += row("Average land price", "model average", false, p0[0], p0[1], "base", inr(ex.base_value), "flat");

        steps.forEach(function (s, i) {
            var p = pos(points[i + 1].lo, points[i + 1].hi);
            var up = s.impact >= 0;
            html += row(s.label, s.display_value, s.assumed, p[0], p[1], up ? "up" : "down",
                inr(s.impact, true), up ? "up" : "down");
        });

        var pf = pos(floor, ex.predicted_price);
        html += row("Predicted price", "per sq.ft", false, pf[0], pf[1], "final", inr(ex.predicted_price), "flat");
        return html;
    }

    function categoryHtml(ex) {
        var cats = ex.categories || [];
        var max = 0;
        cats.forEach(function (c) { max = Math.max(max, Math.abs(c.impact)); });
        max = max || 1;
        return cats.map(function (c) {
            var up = c.impact >= 0;
            var w = Math.abs(c.impact) / max * 50;
            var left = up ? 50 : 50 - w;
            return row(c.category, "", false, left, w, up ? "up" : "down", inr(c.impact, true), up ? "up" : "down");
        }).join("");
    }

    function skeleton(container) {
        injectCss();
        container.innerHTML = '<div class="why-card"><div class="why-skel" style="width:40%"></div>' +
            '<div class="why-skel"></div><div class="why-skel"></div><div class="why-skel" style="width:80%"></div></div>';
    }

    function render(container, ex) {
        if (!container) { return; }
        injectCss();

        if (!ex || !ex.available) {
            container.innerHTML = ex && ex.message
                ? '<div class="why-card"><div class="why-sub">Explanation not available: ' + esc(ex.message) + "</div></div>"
                : "";
            return;
        }

        container.innerHTML =
            '<div class="why-card">' +
            '<div class="why-head"><h3>Why this price?</h3>' +
            '<div class="why-toggle"><button type="button" class="on" data-view="top">Top features</button>' +
            '<button type="button" data-view="cat">By category</button></div></div>' +
            '<div class="why-sub">How each factor moved the estimate, in \u20B9 per sq.ft</div>' +
            '<div class="why-summary">' + esc(ex.summary) + "</div>" +
            '<div class="why-chart"></div>' +
            '<div class="why-note">' + esc(ex.note) + " Items tagged <b>assumed</b> were auto-filled from area averages.</div>" +
            "</div>";

        var chart = container.querySelector(".why-chart");
        var buttons = container.querySelectorAll(".why-toggle button");

        function show(view) {
            chart.innerHTML = view === "cat" ? categoryHtml(ex) : waterfallHtml(ex);
            buttons.forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-view") === view); });
        }

        buttons.forEach(function (b) {
            b.addEventListener("click", function () { show(b.getAttribute("data-view")); });
        });
        show("top");
    }

    window.LandWhy = {render: render, skeleton: skeleton};
})();
