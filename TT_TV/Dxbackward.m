function ux=Dxbackward(u,BoundaryCondition)
%������D_x^-, Ĭ��ѭ���߽�
% ux=imfilter(u,[-1,1,0],BoundaryCondition);
%ʾ����ux=Dxbackward(u,'circular');
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        ux=[u(:,1)-u(:,end) diff(u,1,2)];
    case 'symmetric'
        ux=[zeros(size(u,1),1) diff(u,1,2)];
    case 'zero'
        ux=[u(:,1) diff(u,1,2)];
end
end