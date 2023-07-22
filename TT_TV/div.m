function [u]=div(px,py,BoundaryCondition)
%�������ݶ����ӵĸ���������ɢ��, Ĭ��ѭ���߽�
%u=D_x^-(px)+D_y^-(py);
%ʾ��: u=div(px,py,'circular');
if nargin<3
    BoundaryCondition='circular';
end
u=Dxbackward(px,BoundaryCondition)+Dybackward(py,BoundaryCondition);
end