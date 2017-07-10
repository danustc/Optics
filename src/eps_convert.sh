for file in *.eps; do
    convert -density 100 "$file" "${file%.eps}.png" 
done
