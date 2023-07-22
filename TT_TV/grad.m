function [ux,uy]=grad(u,BoundaryCondition)
%�������ݶ�, Ĭ��ѭ���߽�
%ux=D_x^+(u);uy=D_y^+(u);
%ʾ����[ux,uy]=grad(u,'circular');
if nargin<2
    BoundaryCondition='circular';
end
ux=Dxforward(u,BoundaryCondition);
uy=Dyforward(u,BoundaryCondition);
end