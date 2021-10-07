y=${3:-32}
x=$((3*$y/2))

ss=${4:-00:00:00}

# ffmpeg -ss $ss -i "$1" -to $to -vf "scale=$x:$y:force_original_aspect_ratio=decrease,pad=$x:$y:-1:-1:color=black" -an "$2"

ffmpeg -i "$1" -ss $ss -vf "scale=$x*1.2:$y*1.2:force_original_aspect_ratio=decrease,crop=$x:$y,pad=$x:$y:-1:-1:color=black" -an "$2"

