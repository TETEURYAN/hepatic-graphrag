# Force XeLaTeX for this project regardless of system-wide latexmk defaults
# (e.g. /etc/LatexMk may set $pdf_mode = 4, i.e. lualatex) or of the invoking
# tool's own flags (VimTeX's latexmk call here passes no explicit engine flag).
# A project-local .latexmkrc takes precedence over both, so `latexmk pre.tex`
# and VimTeX's :VimtexCompile always use xelatex without any extra config.
$pdf_mode = 5; # 5 = xelatex
