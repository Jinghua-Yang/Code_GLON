function ux=Dxforward(u,BoundaryCondition)
%������D_x^+, Ĭ��ѭ���߽�
% ux=imfilter(u,[0,-1,1],BoundaryCondition);
%ʾ����ux=Dxforward(u,'circular');
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        ux=[diff(u,1,2) u(:,1)-u(:,end)];
    case 'symmetric'
        ux=[diff(u,1,2) zeros(size(u,1),1)];
    case 'zero'
        ux=[diff(u,1,2) -u(:,end)];
end
end