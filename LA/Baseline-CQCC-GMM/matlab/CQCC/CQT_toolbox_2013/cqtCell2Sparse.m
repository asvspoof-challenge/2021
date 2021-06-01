function cSparse = cqtCell2Sparse(c,M)

bins = size(M,1)/2 - 1;
spLen = M(bins+1);
cSparse = zeros(bins,spLen);

M = M(1:bins+1);
step = 1;
distinctHops = log2(M(bins+1)/M(2))+1;
curNumCoef = M(bins+1);

for ii=1:distinctHops
    idx = [(M == curNumCoef); false];
    temp = cell2mat( c(idx).' ).';
    cSparse(idx,1:step:end) = temp;
    step = step*2;
    curNumCoef = curNumCoef / 2;
end

cSparse = sparse(cSparse);
