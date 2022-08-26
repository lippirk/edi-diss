.PHONY: watch
watch:
	while inotifywait -r . -e modify || true; do                 \
		sleep 1;                                                   \
		pdflatex -synctex=1 -shell-escape -halt-on-error -interaction=nonstopmode thesis.tex; \
	done

.PHONY: bib
bib:
	rm -f *.lof *.log *.lot *.toc *.aux *.bcf *.bbl;
	pdflatex -synctex=1 -shell-escape -interaction=nonstopmode thesis.tex
	bibtex thesis
	pdflatex -synctex=1 -shell-escape -interaction=nonstopmode thesis.tex
	pdflatex -synctex=1 -shell-escape -interaction=nonstopmode thesis.tex
