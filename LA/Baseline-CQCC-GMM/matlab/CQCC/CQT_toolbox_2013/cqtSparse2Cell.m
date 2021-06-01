function cCell = cqtSparse2Cell(cSparse,M,cDC,cNyq)

bins = size(M,1)/2 - 1;
cCell = cell(1,bins+2);
cCell{bins+2} = cNyq;

M = M(1:bins+1);
step = 1;
cSparse = full(cSparse);
distinctHops = log2(M(bins+1)/M(2))+1;
curNumCoef = M(bins+1);

for ii=1:distinctHops
   idx = (M == curNumCoef); 
   temp = cSparse(idx,1:step:end).';
   temp = num2cell(temp,1);
   cCell(idx) = temp;
   step = step*2;
   curNumCoef = curNumCoef / 2;
end

cCell{1} = cDC;