cd RunFRT
for file in ./*sbatch
do
	sbatch "$file"
done