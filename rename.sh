for file in *.*; do
    ext="${file##*.}"
    base="${file%.*}"
    newbase=$(echo "$base" | sed -E 's/ *\([0-9_ :A-Z]*\)$//')
    newname="${newbase}.${ext}"
    if [ "$file" != "$newname" ]; then
        echo "Renaming: '$file' → '$newname'"
        mv "$file" "$newname"
    fi
done
