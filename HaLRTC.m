%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADM algorithm: tensor completion
% paper: Tensor completion for estimating missing values in visual data
% date: 05-22-2011
% min_X: \sum_i \alpha_i \|X_{i(i)}\|_*
% s.t.:  X_\Omega = B_\Omega
% by Ji Liu 
%   @article{Liu2013PAMItensor,
%   title={Tensor completion for estimating missing values in visual data},
%   author={Liu, Ji and Musialski, Przemyslaw and Wonka, Peter and Ye, Jieping},
%   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
%   volume={35},
%   number={1},
%   pages={208--220},
%   year={2013},
% }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, Out] = HaLRTC( data, known, opts )

if isfield(opts, 'tol');         tol      = opts.tol;       end
if isfield(opts, 'maxit');       maxit    = opts.maxit;     end
if isfield(opts, 'rho');         rho      = opts.rho;       end
if isfield(opts, 'beta');        beta     = opts.beta;      end
if isfield(opts, 'alpha');       alpha    = opts.alpha;     end
if isfield(opts, 'max_beta');    max_beta = opts.max_beta;  end

dim = size(opts.X0);

A   = zeros( size(image2vdt256(opts.X0)) );
B   = zeros( size(image2vdt256(opts.X0)) );
A(known) = data;
B(known) = 1;
A = vdt2image256(A);
B = vdt2image256(B);
known = find(B==1);
data  = A(known);


X   = initialization_M( dim, known, data );

Phi = cell(ndims(X), 1);
M = Phi;

for i = 1:ndims(X)
    M{i} = X;
    Phi{i} = zeros(dim);
end

Msum = zeros(dim);
Psum = zeros(dim);
Out.Res=[];Out.ResT=[]; Out.PSNR=[];
for k = 1: maxit
    
    %% update M
    Psum = 0*Psum;
    Msum = 0*Msum;
    for i = 1:ndims(X)
        M{i} = Fold(Pro2TraceNorm(Unfold(X-Phi{i}/beta, dim, i), alpha(i)/beta), dim, i);
        Psum = Psum + Phi{i};
        Msum = Msum + M{i};
    end
    
    %% update X    
    Xold = X;
    X = (Psum + beta*Msum) / (ndims(X)*beta);
    X(known) = data;
    X = min( 255, max( X, 0 ));
    
    %% update Phi
    for i = 1:ndims(X)
        Phi{i} = Phi{i} + beta*(M{i} - X);
    end
    
   beta = min(beta * rho, max_beta);
    %% check the convergence
    if isfield(opts, 'Xtrue')
       XT=opts.Xtrue;    
       resT=norm(X(:)-XT(:))/norm(XT(:)); 
       psnr=PSNR(X,XT);
       Out.ResT = [Out.ResT,resT];
       Out.PSNR = [Out.PSNR,psnr];
    end
    
    res=norm(X(:)-Xold(:))/norm(Xold(:));
    Out.Res = [Out.Res,res];
    
    if res < tol 
        break;
    end 
end
end