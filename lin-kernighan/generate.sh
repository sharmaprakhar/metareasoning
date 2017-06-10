for file in instances/*
do
  ./LKSolver < "$file" > results/"$file"
done 
