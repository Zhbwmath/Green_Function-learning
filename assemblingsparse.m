function [A,b,freeNodes] = assemblingsparse(node,elem,Db,pde)
N=size(node,1);
NT=size(elem,1);
ii = zeros(9*NT,1);jj=zeros(9*NT,1);sA1 = zeros(9*NT,1);sA2 = zeros(9*NT,1);
ve(:,:,3) = node(elem(:,2),:)-node(elem(:,1),:);
ve(:,:,1) = node(elem(:,3),:)-node(elem(:,2),:);
ve(:,:,2) = node(elem(:,1),:)-node(elem(:,3),:);
% each layer of ve(:,:,k) represents the difference, (xi-xj,yi-yj),i,j given
area = 0.5*abs(-ve(:,1,3).*ve(:,2,2)+ve(:,2,3).*ve(:,1,2));
% area calculated by vector product, size:(NT,1)
index = 0;
for i=1:3
    for j=1:3 
        ii(index+1:index+NT) = elem(:,i);
        jj(index+1:index+NT) = elem(:,j);
        sA1(index+1:index+NT) = dot(ve(:,:,i),ve(:,:,j),2)./(4*area);
        % <ni,nj> = <xi,xj>, < , > for inner product
        if(i==j)
            sA2(index+1:index+NT) =  2.*area.*ones(NT,1)./12;
        else
            sA2(index+1:index+NT) =  area.*ones(NT,1)./12;
        end
        index = index +NT;
    end
end
A = sparse(ii,jj,sA1,N,N);
%M = sparse(ii,jj,sA2,N,N);
fixedNode = unique(Db);
freeNodes = setdiff(1:N,fixedNode);
%% Right hand side
mid1 = (node(elem(:,2),:)+node(elem(:,3),:))/2;
mid2 = (node(elem(:,3),:)+node(elem(:,1),:))/2;
mid3 = (node(elem(:,1),:)+node(elem(:,2),:))/2;
bt1 = area.*(pde.f(mid2)+pde.f(mid3))/6;
bt2 = area.*(pde.f(mid3)+pde.f(mid1))/6;
bt3 = area.*(pde.f(mid1)+pde.f(mid2))/6;
% 3-point, 2-order format
b = accumarray(elem(:),[bt1;bt2;bt3],[N 1]);
end
