function c = cqtFillSparse(c,M,B)

%Repeat coefficients in sparse matrix until the next valid coefficient. 
%For visualization this is an overkill since we could image each CQT bin
%seperately, however, in some case this might come in handy.

bins = size(c,1);
M = M(1:bins);
distinctHops = log2(M(bins)/M(2))+1;

curNumCoef = M(end-1) / 2;
step = 2;
for ii=1:distinctHops -1  
    idx = (M == curNumCoef);
    temp = c(idx,1:step:end);
    temp = repmat(temp,step,1);
    temp = reshape(temp(:), nnz(idx), []);
    c(idx,:) = temp;
    step = 2*step;
    curNumCoef = curNumCoef / 2;
end



