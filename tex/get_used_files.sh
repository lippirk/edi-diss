#!/bin/bash
grep -E 'includegraphics.*\.(pdf|png|jpg)' thesis.tex | sed -Ee 's/.*\{([^\}]*?).*/\1/g'
grep -E 'input.*\.(tex|txt)' thesis.tex               | sed -Ee 's/.*input\{([^\}]*?)\}.*/\1/g'

## ./get_used_files.sh | xargs git add
