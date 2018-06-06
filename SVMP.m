function output=SVMP(positive,negative)
label=[ones(size(positive,1),1);-ones(size(negative,1),1)];
feature=sparse(double([positive;negative]));
model=train(label,feature,'-s 1 -c 1  -q'); %Parameter C here is fixed as 1
output=model.w;
end