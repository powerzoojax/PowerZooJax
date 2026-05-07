function upgradeMermaidBlocks() {
  document.querySelectorAll("pre.mermaid").forEach((block) => {
    const graph = document.createElement("div");
    graph.className = "mermaid";
    graph.textContent = block.textContent;
    block.replaceWith(graph);
  });
}

function tagMermaidDiagramKinds(nodes) {
  nodes.forEach((node) => {
    const svg = node.querySelector("svg");
    const role = svg && svg.getAttribute("aria-roledescription");
    const isClassDiagram =
      Boolean(node.querySelector("g.classGroup")) || role === "class";
    node.classList.toggle("mermaid-class-diagram", isClassDiagram);
  });
}

// Mermaid v11 emits class diagrams (and some others) with width="100%"
// + inline `max-width:<viewBox-width>px`. The 100% width forces the SVG to
// fill the container, which scales the logical 13px font up to ~22-40px.
// Pin the SVG width to its viewBox width so it renders at its natural
// size, matching how flowcharts already look.
function unstretchMermaidSvgs(nodes) {
  nodes.forEach((node) => {
    const svg = node.querySelector("svg");
    if (!svg) return;
    if (svg.getAttribute("width") === "100%") {
      const vb = svg.getAttribute("viewBox");
      let pinned = false;
      if (vb) {
        const parts = vb.trim().split(/\s+/);
        const w = parseFloat(parts[2]);
        if (Number.isFinite(w) && w > 0) {
          svg.setAttribute("width", String(w));
          pinned = true;
        }
      }
      if (!pinned) {
        svg.removeAttribute("width");
      }
    }
    if (svg.style && svg.style.maxWidth) {
      svg.style.maxWidth = "";
    }
  });
}

function getMermaidConfig() {
  const scheme = document.body?.getAttribute("data-md-color-scheme");
  const darkMode = scheme === "slate";
  const baseFontFamily =
    "Roboto, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
  const baseTypography = {
    fontFamily: baseFontFamily,
    fontSize: "13px",
  };
  const diagramTheme = darkMode
    ? {
        ...baseTypography,
        primaryColor: "#143f3b",
        primaryTextColor: "#ecfeff",
        primaryBorderColor: "#5eead4",
        secondaryColor: "#143f3b",
        secondaryTextColor: "#ecfeff",
        secondaryBorderColor: "#5eead4",
        tertiaryColor: "#143f3b",
        tertiaryTextColor: "#ecfeff",
        tertiaryBorderColor: "#5eead4",
        textColor: "#ecfeff",
        lineColor: "#5eead4",
        defaultLinkColor: "#5eead4",
        edgeLabelBackground: "#0f172a",
        clusterBkg: "#0d2f2c",
        clusterBorder: "#2dd4bf",
        titleColor: "#ecfeff",
        mainBkg: "#143f3b",
        nodeBorder: "#5eead4",
      }
    : {
        ...baseTypography,
        primaryColor: "#e8f7f4",
        primaryTextColor: "#123c3a",
        primaryBorderColor: "#0f766e",
        secondaryColor: "#e8f7f4",
        secondaryTextColor: "#123c3a",
        secondaryBorderColor: "#0f766e",
        tertiaryColor: "#e8f7f4",
        tertiaryTextColor: "#123c3a",
        tertiaryBorderColor: "#0f766e",
        textColor: "#123c3a",
        lineColor: "#0f766e",
        defaultLinkColor: "#0f766e",
        edgeLabelBackground: "#ffffff",
        clusterBkg: "#f6fbfa",
        clusterBorder: "#9bd5cf",
        titleColor: "#123c3a",
        mainBkg: "#e8f7f4",
        nodeBorder: "#0f766e",
      };

  return {
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    themeVariables: diagramTheme,
    flowchart: {
      htmlLabels: true,
      useMaxWidth: false,
    },
    classDiagram: {
      hideEmptyMembersBox: true,
      htmlLabels: true,
      useMaxWidth: false,
    },
    class: {
      hideEmptyMembersBox: true,
      htmlLabels: true,
      useMaxWidth: false,
    },
  };
}

document$.subscribe(async () => {
  if (!window.mermaid) {
    return;
  }

  upgradeMermaidBlocks();
  window.mermaid.initialize(getMermaidConfig());

  const nodes = Array.from(document.querySelectorAll("div.mermaid")).filter(
    (node) => node.dataset.mermaidRendered !== "true",
  );
  if (nodes.length === 0) {
    return;
  }

  nodes.forEach((node, index) => {
    if (!node.id) {
      node.id = `mermaid-${Date.now()}-${index}`;
    }
  });

  try {
    await window.mermaid.run({ nodes });
    tagMermaidDiagramKinds(nodes);
    unstretchMermaidSvgs(nodes);
    nodes.forEach((node) => {
      node.dataset.mermaidRendered = "true";
    });
  } catch (err) {
    console.warn("Mermaid render:", err);
  }
});
