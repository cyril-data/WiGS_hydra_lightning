cd RunSimulations
for file in ./*sbatch
do
	sbatch "$file"
done