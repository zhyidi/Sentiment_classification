#i=0
#while [ "${i}" != "5" ]
#do
#	i=$(($i+1))
#	~/.conda/envs/pytorch/bin/python ~/Graduation_Project/bayes.py > ~/Graduation_Project/Result/bayes_{$i}.txt 2>&1
#done

#i=0
#while [ "${i}" != "5" ]
#do
#	i=$(($i+1))
#       ~/.conda/envs/pytorch/bin/python ~/Graduation_Project/svm.py > ~/Graduation_Project/Result/svm_{$i}.txt 2>&1
#done

#i=0
#while [ "${i}" != "5" ]
#do
#	i=$(($i+1))
#       ~/.conda/envs/pytorch/bin/python ~/Graduation_Project/XGboost.py > ~/Graduation_Project/Result/XGboost_{$i}.txt 2>&1
#done

i=0
while [ "${i}" != "5" ]
do
	i=$(($i+1))
       ~/.conda/envs/pytorch/bin/python ~/Graduation_Project/lstm.py > ~/Graduation_Project/Result/lstm_{$i}.txt 2>&1
done

#i=0
#while [ "${i}" != "5" ]
#do
#	i=$(($i+1))
#        ~/.conda/envs/pytorch/bin/python ~/Graduation_Project/bert.py > ~/Graduation_Project/Result/bert_{$i}.txt 2>&1
#done

#i=0
#while [ "${i}" != "5" ]
#do
#	i=$(($i+1))
#        ~/.conda/envs/pytorch/bin/python ~/Graduation_Project/bert_lstm.py > ~/Graduation_Project/Result/bert_lstm_{$i}.txt 2>&1
#done
