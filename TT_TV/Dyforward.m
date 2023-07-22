function uy=Dyforward(u,BoundaryCondition)
%������D_y^+, Ĭ��ѭ���߽�
% uy=imfilter(u,[0;-1;1],BoundaryCondition);
%ʾ����uy=Dyforward(u,'circular');
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        uy=[diff(u,1,1);u(1,:)-u(end,:)];
    case 'symmetric'
        uy=[diff(u,1,1);zeros(1,size(u,2))];
    case 'zero'
        uy=[diff(u,1,1);-u(end,:)];
end
end