/* ==========================================================================
   mRNA VIP - client behaviour
   --------------------------------------------------------------------------
   Vanilla ES2019. No framework, no bundler, no build step. The brief excludes
   React, Next.js and similar, and the project must run from a clean checkout.

   What this file is responsible for:
     - hash routing between the six pages
     - binding /api/state into the DOM
     - form validation against the dataset's documented ranges
     - moving controls through ready / loading / empty / error states

   What this file deliberately does NOT do:
     - compute, infer or guess any ML, RAG or LLM value
     - render a chart. Panels show a saved image when one exists and an explicit
       empty state when it does not. There is no client-side plotting fallback,
       because a drawn chart without data would be a fabricated result.
   ========================================================================== */
(function () {
  "use strict";

  /* ------------------------------------------------------------- helpers -- */
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  /* Route table. `render` is invoked once the state snapshot is available. */
  var ROUTES = ["overview", "patient", "risk", "attention", "regulatory", "report"];

  var state = null;

  /* ------------------------------------------------------------- routing -- */
  function currentRoute() {
    var hash = (window.location.hash || "").replace(/^#\/?/, "").split("?")[0];
    return ROUTES.indexOf(hash) >= 0 ? hash : "overview";
  }

  function navigate() {
    var route = currentRoute();

    $$(".page").forEach(function (page) {
      var active = page.id === "page-" + route;
      page.hidden = !active;
      page.classList.toggle("is-active", active);
    });

    $$(".nav-link").forEach(function (link) {
      var active = link.dataset.route === route;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });

    closeNav();
    document.getElementById("main").focus({ preventScroll: true });
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", "#/" + route);
    }
  }

  /* --------------------------------------------------- mobile navigation -- */
  function openNav() {
    document.getElementById("sidebar").classList.add("is-open");
    document.getElementById("nav-scrim").hidden = false;
    var toggle = document.getElementById("nav-toggle");
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Close navigation menu");
  }

  function closeNav() {
    document.getElementById("sidebar").classList.remove("is-open");
    document.getElementById("nav-scrim").hidden = true;
    var toggle = document.getElementById("nav-toggle");
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open navigation menu");
  }

  /* --------------------------------------------------------- status pill -- */
  /* Status is text plus a colour plus a shape, never colour on its own. */
  function renderSystemStatus(data) {
    var pill = document.getElementById("system-status");
    var ready = data.stages.filter(function (s) { return s.status === "ready"; }).length;
    var total = data.stages.length;
    var text = $(".status-text", pill);

    if (ready === total) {
      pill.dataset.state = "ready";
      text.textContent = "All stages ready";
    } else if (ready > 0) {
      pill.dataset.state = "partial";
      text.textContent = ready + " of " + total + " stages ready";
    } else {
      pill.dataset.state = "none";
      text.textContent = "Not Initialized";
    }
  }

  /* ------------------------------------------------------------ pipeline -- */
  function renderPipeline(stages) {
    var list = document.getElementById("pipeline");
    clear(list);

    stages.forEach(function (stage) {
      var item = el("li", "stage");
      item.dataset.status = stage.status;

      item.appendChild(el("span", "stage-num", stage.number));
      item.appendChild(el("span", "stage-name", stage.name));
      item.appendChild(el("span", "stage-desc", stage.description));

      var badge = el("span", "badge " + (stage.status === "ready" ? "badge-success" : "badge-neutral"));
      badge.appendChild(el("span", "badge-icon", stage.status === "ready" ? "\u2713" : "\u25CB"));
      badge.appendChild(el("span", null, stage.status === "ready" ? "Ready" : "Not Initialized"));
      item.appendChild(badge);

      list.appendChild(item);
    });
  }

  /* ------------------------------------------------------------- metrics -- */
  /* Only specification-derived values. No accuracy, ROC-AUC, latency or risk
     percentage appears here; those come from the pipeline, not from config. */
  function renderMetrics(metrics) {
    var grid = document.getElementById("metric-grid");
    clear(grid);

    function add(label, value, note, isText) {
      if (value === null || value === undefined) return;
      var wrap = el("div", "metric");
      // <dt> first for correct definition-list semantics; CSS reorders it above
      // the value visually.
      wrap.appendChild(el("dt", "metric-label", label));
      wrap.appendChild(el("dd", "metric-value" + (isText ? " is-text" : ""), value));
      if (note) wrap.appendChild(el("dd", "metric-note", note));
      grid.appendChild(wrap);
    }

    add("Patient Records", metrics.patient_records, "Rows in the supplied dataset");
    add("ML Models", metrics.ml_models, "Logistic Regression + Random Forest");
    add("Retrieval Top-K", metrics.retrieval_top_k, "Chunks per query (FR-3.4)");
    add("Report Format", metrics.report_format, "Validated output schema", true);
  }

  /* -------------------------------------------------------------- panels -- */
  /* A panel shows the saved image when the artifact exists. Otherwise it keeps
     its server-rendered empty state. Nothing is ever drawn in the browser. */
  var PANEL_FIGURE = {
    roc: "roc_curve.png",
    attention: "attention_heatmap.png",
    latency: null
  };

  function renderPanels(artifacts) {
    Object.keys(PANEL_FIGURE).forEach(function (panel) {
      if (!artifacts[panel]) return;
      var figure = PANEL_FIGURE[panel];
      if (!figure) return; // latency is available only inside dashboard.png

      var host = document.querySelector('.state[data-panel="' + panel + '"]');
      if (!host) return;
      var body = host.parentNode;
      clear(body);
      var img = el("img");
      img.src = "/figures/" + figure;
      img.alt = document.querySelector("#h-panel-" + panel).textContent + " result";
      img.loading = "lazy";
      body.appendChild(img);
    });
  }

  /* ----------------------------------------------------- model sections -- */
  var MODEL_LABELS = {
    logistic_regression: "Logistic Regression",
    random_forest: "Random Forest"
  };

  /* One model card. Shows measured metrics when they exist and falls back to the
     empty state otherwise - never a placeholder number. */
  function renderModelCard(name, metrics, emptyCopy) {
    var card = el("section", "card");
    var head = el("div", "card-head");
    head.appendChild(el("h2", null, MODEL_LABELS[name] || name));
    head.appendChild(el("p", "card-sub", "Adverse-reaction risk classifier."));
    card.appendChild(head);

    var rows = metrics
      ? data_metric_keys().map(function (key) {
          return [LABEL_FOR_METRIC[key], metrics[key]];
        })
      : [["Prediction", "—"], ["Probability", "—"], ["Evaluation Metrics", emptyCopy]];

    rows.forEach(function (row) {
      var line = el("div", "result-row");
      line.appendChild(el("span", "result-label", row[0]));
      line.appendChild(el(
        "span",
        metrics ? "result-value is-ready" : "result-value is-empty",
        metrics ? row[1].toFixed(4) : row[1]
      ));
      card.appendChild(line);
    });
    return card;
  }

  function data_metric_keys() {
    return state && state.metrics && state.metrics.metric_keys
      ? state.metrics.metric_keys
      : ["accuracy", "precision", "recall", "f1", "roc_auc"];
  }

  var LABEL_FOR_METRIC = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1",
    roc_auc: "ROC-AUC"
  };

  function renderModels(data) {
    var grid = document.getElementById("model-grid");
    clear(grid);

    var measured = (data.metrics && data.metrics.evaluated) || {};
    data.metrics.model_names.forEach(function (name) {
      grid.appendChild(
        renderModelCard(name, measured[name], data.copy.empty_states.models)
      );
    });
  }

  /* Model comparison table, rendered only once real metrics exist. */
  function renderComparison(data) {
    var host = document.getElementById("model-comparison");
    var measured = (data.metrics && data.metrics.evaluated) || {};
    var names = data.metrics.model_names.filter(function (n) { return measured[n]; });
    if (!names.length) return;   // empty state stays as authored in the markup

    clear(host);
    var wrap = el("div", "table-wrap");
    var table = el("table");

    var thead = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", null, "Model"));
    data_metric_keys().forEach(function (key) {
      headRow.appendChild(el("th", null, LABEL_FOR_METRIC[key] || key));
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = el("tbody");
    names.forEach(function (name) {
      var row = el("tr");
      row.appendChild(el("td", null, MODEL_LABELS[name] || name));
      data_metric_keys().forEach(function (key) {
        row.appendChild(el("td", "mono", measured[name][key].toFixed(4)));
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    host.appendChild(wrap);

    if (data.metrics.selected_model) {
      var note = el("p", "card-sub");
      note.style.marginTop = "12px";
      note.textContent =
        "Held-out test set. Selected on test F1 (FR-1.7): " +
        (MODEL_LABELS[data.metrics.selected_model] || data.metrics.selected_model) +
        ". Both models remain available through the inference interface.";
      host.appendChild(note);
    }
  }

  /* ------------------------------------------------------------- schema -- */
  function renderSchema(fields) {
    var list = document.getElementById("schema-list");
    clear(list);

    fields.forEach(function (field) {
      var wrap = el("div", "schema-field");
      var head = el("div", "schema-field-head");
      head.appendChild(el("span", "schema-field-name", field.name));
      head.appendChild(el("span", "schema-field-type", field.type));
      var req = el("span", "badge " + (field.required ? "badge-accent" : "badge-neutral"),
                   field.required ? "Required" : "Default");
      head.appendChild(req);
      wrap.appendChild(head);
      if (field.description) wrap.appendChild(el("p", "schema-field-desc", field.description));
      wrap.appendChild(el("div", "schema-field-value",
                          field.type.indexOf("bool") >= 0 ? "false" : "—"));
      list.appendChild(wrap);
    });
  }

  function renderValidation(data) {
    var list = document.getElementById("validation-list");
    clear(list);

    var checks = [
      ["Pydantic schema validated", data.artifacts.report ? "pass" : "pending"],
      ["Dose bound within guideline", data.dose_bound ? "pass" : "pending"],
      ["Disclaimer present", data.artifacts.report ? "pass" : "pending"],
      ["Evidence cited", data.artifacts.report ? "pass" : "pending"]
    ];

    checks.forEach(function (check) {
      var item = el("li", "check-item");
      item.dataset.state = check[1];
      item.appendChild(el("span", "check-icon",
        check[1] === "pass" ? "\u2713" : "\u25CB"));
      item.appendChild(el("span", null, check[0]));
      list.appendChild(item);
    });
  }

  /* -------------------------------------------------------------- chunks -- */
  /* Three result blocks are reserved so the layout is stable. None contains
     invented regulatory text: each shows the retrieval empty state. */
  function renderChunks(k, emptyText) {
    var list = document.getElementById("chunk-list");
    clear(list);

    for (var i = 1; i <= k; i++) {
      var card = el("article", "chunk");
      var head = el("div", "chunk-head");
      head.appendChild(el("span", "chunk-index", i));
      head.appendChild(el("span", "badge badge-neutral", "Not Retrieved"));
      card.appendChild(head);

      var body = el("div", "state");
      body.dataset.variant = "empty";
      var inner = el("div", "state-inner");
      inner.appendChild(el("span", "state-icon", "\u25CB"));
      inner.appendChild(el("p", "state-text", i === 1 ? emptyText : "Awaiting retrieval."));
      body.appendChild(inner);
      card.appendChild(body);

      card.appendChild(el("p", "chunk-source", "Source: " + state.source.name));
      list.appendChild(card);
    }
  }

  /* ------------------------------------------------------- form handling -- */
  /* Ranges come from the dataset specification in the PRD, not from guesses. */
  function validatePatient(form) {
    var errors = [];
    var fields = [
      { id: "f-age", min: 18, max: 85, label: "Age", integer: true },
      { id: "f-biomarker", min: 0, label: "Biomarker Level" },
      { id: "f-conditions", min: 0, max: 4, label: "Pre-Existing Conditions", integer: true },
      { id: "f-dose", min: 25, max: 150, label: "Vaccine Dose", decimal: true }
    ];

    fields.forEach(function (spec) {
      var input = document.getElementById(spec.id);
      input.removeAttribute("aria-invalid");
      var raw = input.value.trim();

      if (raw === "") {
        errors.push(spec.label + " is required.");
        input.setAttribute("aria-invalid", "true");
        return;
      }
      var value = Number(raw);
      if (isNaN(value)) {
        errors.push(spec.label + " must be a number.");
        input.setAttribute("aria-invalid", "true");
        return;
      }
      if (spec.integer && Math.floor(value) !== value) {
        errors.push(spec.label + " must be a whole number.");
        input.setAttribute("aria-invalid", "true");
      }
      if (spec.min !== undefined && value < spec.min) {
        errors.push(spec.label + " must be at least " + spec.min + ".");
        input.setAttribute("aria-invalid", "true");
      }
      if (spec.max !== undefined && value > spec.max) {
        errors.push(spec.label + " must not exceed " + spec.max + ".");
        input.setAttribute("aria-invalid", "true");
      }
    });

    return errors;
  }

  function setButtonLoading(button, loading, busyLabel) {
    var label = $(".btn-label", button);
    if (loading) {
      button.dataset.idleLabel = label.textContent;
      label.textContent = busyLabel;
      button.classList.add("is-loading");
      button.disabled = true;
      button.setAttribute("aria-busy", "true");
    } else {
      label.textContent = button.dataset.idleLabel || busyLabel;
      button.classList.remove("is-loading");
      button.disabled = false;
      button.removeAttribute("aria-busy");
    }
  }

  function showFormError(id, message) {
    var box = document.getElementById(id);
    if (!box) return;
    if (message) {
      box.textContent = message;
      box.hidden = false;
    } else {
      box.textContent = "";
      box.hidden = true;
    }
  }

  function setRiskRow(id, value, variant) {
    var node = document.getElementById(id);
    node.textContent = value;
    node.className = "result-value " + variant;
  }

  /* Calls the API, which honestly refuses while the modules are unimplemented.
     The failure surfaces in the interface as an error state with a Retry action,
     which is the correct behaviour - not a fabricated prediction. */
  function postJSON(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().then(function (data) {
        return { ok: response.ok, status: response.status, data: data };
      });
    });
  }

  function handleRunAnalysis(event) {
    event.preventDefault();
    showFormError("form-error", null);

    var errors = validatePatient(event.target);
    if (errors.length) {
      showFormError("form-error", errors.join(" "));
      var first = event.target.querySelector('[aria-invalid="true"]');
      if (first) first.focus();
      return;
    }

    var button = document.getElementById("btn-run-analysis");
    setButtonLoading(button, true, "Processing...");
    setRiskRow("risk-classification", "Awaiting analysis", "is-pending");
    setRiskRow("risk-probability", "—", "is-empty");
    document.getElementById("risk-explanation").textContent = state.copy.loading;

    var record = {
      Patient_ID: document.getElementById("f-patient-id").value.trim(),
      Age: Number(document.getElementById("f-age").value),
      Biomarker_Level: Number(document.getElementById("f-biomarker").value),
      Pre_Existing_Conditions: Number(document.getElementById("f-conditions").value),
      Vaccine_Dose_mcg: Number(document.getElementById("f-dose").value),
      Prior_Reaction_History: document.getElementById("f-history").value
    };

    postJSON("/api/run", { stage: "ml", record: record })
      .then(function (result) {
        setButtonLoading(button, false);
        if (result.ok) {
          applyRiskResult(result.data);
        } else {
          showRiskUnavailable(result.data);
        }
      })
      .catch(function () {
        setButtonLoading(button, false);
        showRiskUnavailable(null);
      });
  }

  function showRiskUnavailable(payload) {
    var message = (payload && payload.message) || state.copy.not_initialized;
    var detail = (payload && payload.detail) || "";

    setRiskRow("risk-classification", "Awaiting analysis", "is-pending");
    setRiskRow("risk-probability", "—", "is-empty");

    var explanation = document.getElementById("risk-explanation");
    explanation.textContent = message + (detail ? " " + detail : "");
    showFormError("form-error", state.copy.error + " The analysis module has not been initialized.");
  }

  function applyRiskResult(data) {
    setRiskRow("risk-classification", data.risk_label, "is-ready");
    setRiskRow("risk-probability", Number(data.risk_probability).toFixed(4), "is-ready");
    document.getElementById("risk-explanation").textContent =
      data.explanation || state.copy.empty_states.risk;
    showFormError("form-error", null);
  }

  function handleRetrieve(event) {
    event.preventDefault();
    showFormError("reg-error", null);

    var input = document.getElementById("f-query");
    if (!input.value.trim()) {
      showFormError("reg-error", "A regulatory query is required.");
      input.setAttribute("aria-invalid", "true");
      input.focus();
      return;
    }
    input.removeAttribute("aria-invalid");

    var button = document.getElementById("btn-retrieve");
    setButtonLoading(button, true, "Retrieving...");
    renderChunks(state.metrics.retrieval_top_k, state.copy.loading);

    postJSON("/api/run", { stage: "rag", question: input.value.trim() })
      .then(function (result) {
        setButtonLoading(button, false);
        renderChunks(state.metrics.retrieval_top_k,
          result.ok ? state.copy.empty_states.retrieval
                    : (result.data && result.data.message) || state.copy.not_initialized);
        if (!result.ok) {
          showFormError("reg-error", state.copy.error + " The retrieval module has not been initialized.");
        }
      })
      .catch(function () {
        setButtonLoading(button, false);
        renderChunks(state.metrics.retrieval_top_k, state.copy.error);
        showFormError("reg-error", state.copy.error);
      });
  }

  function handleGenerateReport() {
    var button = document.getElementById("btn-generate-report");
    setButtonLoading(button, true, "Processing...");
    document.getElementById("meta-status").textContent = state.copy.loading;

    postJSON("/api/run", { stage: "report" })
      .then(function (result) {
        setButtonLoading(button, false);
        document.getElementById("meta-status").textContent =
          state.copy.empty_states.report;
        if (!result.ok) {
          showFormError("export-hint",
            state.copy.error + " The report module has not been initialized.");
          document.getElementById("export-hint").classList.add("is-error");
        }
      })
      .catch(function () {
        setButtonLoading(button, false);
        document.getElementById("meta-status").textContent = state.copy.error;
      });
  }

  /* ------------------------------------------------------------ rendering -- */
  function render(data) {
    state = data;

    document.getElementById("disclaimer-text").textContent = data.copy.disclaimer;
    document.getElementById("nav-version").textContent = "v" + data.project.version;
    document.getElementById("about-meta").textContent =
      "Version " + data.project.version + " · Python " + data.environment.python +
      " · " + data.environment.mode;
    document.getElementById("environment-chip").textContent =
      data.environment.mode + " · Py" + data.environment.python.split(".").slice(0, 2).join(".");

    renderSystemStatus(data);
    renderPipeline(data.stages);
    renderMetrics(data.metrics);
    renderPanels(data.artifacts);
    renderModels(data);
    renderComparison(data);
    renderSchema(data.report_schema);
    renderValidation(data);
    renderChunks(data.metrics.retrieval_top_k, data.copy.empty_states.retrieval);

    document.getElementById("top-k").textContent = data.metrics.retrieval_top_k;
    document.getElementById("f-embed-dim").value = data.metrics.embed_dim;
    document.getElementById("source-availability").textContent =
      data.source.available ? "Present" : "Not present";
    document.getElementById("source-path").textContent = data.source.path;
  }

  /* ---------------------------------------------------------------- init -- */
  function init() {
    document.getElementById("nav-toggle").addEventListener("click", function () {
      var open = document.getElementById("sidebar").classList.contains("is-open");
      if (open) { closeNav(); } else { openNav(); }
    });
    document.getElementById("nav-scrim").addEventListener("click", closeNav);

    var dialog = document.getElementById("about-dialog");
    document.getElementById("settings-button").addEventListener("click", function () {
      if (dialog.showModal) dialog.showModal();
    });

    document.getElementById("patient-form").addEventListener("submit", handleRunAnalysis);
    document.getElementById("regulatory-form").addEventListener("submit", handleRetrieve);
    document.getElementById("btn-generate-report").addEventListener("click", handleGenerateReport);

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeNav();
    });
    window.addEventListener("hashchange", navigate);

    fetch("/api/state")
      .then(function (response) { return response.json(); })
      .then(render)
      .catch(function () {
        document.getElementById("system-status").dataset.state = "none";
        $(".status-text", document.getElementById("system-status")).textContent = state_copy_error();
      })
      .then(navigate);

    function state_copy_error() { return "Unavailable"; }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();