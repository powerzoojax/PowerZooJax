// textmacros: makes \text{...} treat \_ \$ \& etc. like LaTeX text mode.
// Without it, \_ inside \text{} shows a visible backslash (MathJax #1770).
window.MathJax = {
  loader: {
    load: ["[tex]/textmacros"],
  },
  tex: {
    packages: { "[+]": ["textmacros"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
  },
  svg: {
    fontCache: "global",
  },
};
