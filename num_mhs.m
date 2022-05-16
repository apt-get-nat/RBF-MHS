function [Bp,rs,Bp_unmodified,Bfullset] = num_mhs(dens,pres, Bx0,By0,Bz0, n, ...
        nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, g,...
        index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
        gamma, varargin)
% num_mhs.mat
% Author: Nathaniel H. Mathews
% Input:
%        dens - the density profile of the plasma within the volume
%        pres - the pressure profile of the plasma
%        Bx0  - a vector of dirichlet boundary conditions on Bx
%        By0  - a vector of dirichlet boundary conditions on By
%        Bz0  - a vector of dirichlet boundary conditions on Bz
%        n    - the size of the computational grid, including boundaries 
%               and the ghost nodes
%       nodes - nx3 matrix of gridpoint locations in 3D
%        Dx   - the differentiation matrix operating on x
%        Dy   - the differentiation matrix operating on y
%        Dz   - the differentiation matrix operating on z (vertical)
%        Dxx  - the second derivative matrix operating on x
%        Dyy  - the second derivative matrix operating on y
%        Dzz  - the second derivative matrix operating on z
%        g    - the gravitational constant as scaled in dens and pres.
%    index_x0 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on one of the side
%               boundaries
%    index_x1 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on the opposite
%               boundary
%    index_y0 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on another side
%               boundary
%    index_y1 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on the opposite
%               boundary
%    index_z0 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on the lower
%               boundary
%    index_z1 - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of gridpoints on the upper
%               boundary
%    index_gh - vector of indices that correspond to rows of the nodes
%               matrix that hold locations of the ghost nodes below the
%               lower boundary
%       gamma - the hyperviscocity parameter
% 
%        Optional inputs: num_mhs(..., initField) or
%                         num_mhs(..., initField, preconField)
%   initField - the initial guess for the quasi-newton descent. The default
%               is a potential field model solution
% preconField - the field used as a preconditioner. The default is
%               the default or passed-in value of initField.
%
% Output:
%        Bp   - The final magnetic field output. This is the answer.
%        rs   - MHS residual at each QN step. Useful for diagnosing
%               convergence.
% Bp_unmodified - The QN solution before divergence cleaning was applied.
%               May more strictly align with MHS balance, but at the cost
%               of nonzero divergence.
%    Bfullset - A full matrix of the magnetic field from every step of the
%               QN method. Useful for diagnosing convergence.



if nargin == 22
    [Bpx, Bpy, Bpz] = potfield(nodes,index_z0,index_z1,...
            Bz0,Dx,Dy,Dz,Dxx,Dyy,Dzz);
    Bp = [Bpx;Bpy;Bpz];
    Bpot = Bp;
elseif nargin == 23
    Bp = varargin{1};
    Bpot = Bp;
elseif nargin == 24
    Bp = varargin{1};
    Bpot = varargin{2};
else
    error('Improper number of inputs.');
end

% Convergence parameters
residual_tol = 1e-3;
maxiters = 20;
relative_tol = 1e-2;


I = speye(n,n);
lap = Dx*Dx+Dy*Dy+Dz*Dz;
lap(index_z1,:) = I(index_z1,:);
lap(index_gh,:) = Dz(index_z0,:);

lap(index_x0,:) = Dx(index_x0,:);
lap(index_x1,:) = Dx(index_x1,:);
lap(index_y0,:) = Dy(index_y0,:);
lap(index_y1,:) = Dy(index_y1,:);

% Preconditioner for lap if necessary for gmres
% fprintf('Building LU...\n');
% options.type = 'ilutp'; options.udiag=1; options.droptol=1e-2;
% [L,U] = ilu(lap, options);
% fprintf('Built.\n');


iters = 0;
rs = [Inf];

res_fn = @(B)resB(B,dens,pres, Bx0,By0,Bz0,n, ...
        Dx, Dy, Dz, Dxx, Dyy, Dzz, g,...
        index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, gamma);
    
res_fn_norm = @(B)(norm(res_fn(B), 2))/sqrt(n);

Bfullset = zeros(numel(Bp),maxiters);
Bfullset(:,1) = Bp;


fprintf('Initial residual: %e\n',res_fn_norm(Bp));

while rs(end) > residual_tol && iters < maxiters && (numel(rs)==1 || abs(rs(end)-rs(end-1))/rs(end)>relative_tol)
    iters = iters+1;
    
    % Find balance
    r = res_fn(Bp);
    lseps = 1;
    
    J = [resBJ(Bp, n, Dx, Dy, Dz, Dxx, Dyy, Dzz, ...
                  index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, gamma)];
    divmat = [Dx,Dy,Dz];
    
    % Do the reverse symmetric cuthill-mckee on J
    cuthill = symrcm(J);
    rOrdered = r(cuthill);
    J = [J(cuthill, cuthill); lseps*divmat(:,cuthill)];
    
    % LSQR (this is the big call)
    precond = @(B,tag)(B-Bpot(cuthill));
    [update,flag,relres,iter,resvec] = lsqr(J, ...
                    [rOrdered;divmat*Bp],1e-5,1e4,precond);
    resvec = resvec/norm(r);
    
    % Undo reverse symmetric cuthill-mckee
    s(cuthill) = 1:3*n;
    update = update(s);
    
    % We perform a QN line search algorithm. Beta is the parameter that
    % controls how far down the update-vector we step.
    beta = 1;
    while res_fn_norm(Bp) - res_fn_norm(Bp-beta*(update)) < 0
        beta = 0.5*beta;
        if abs(beta) < 1e-10
            fprintf(' ! beta break; norm = %e\n',res_fn_norm(Bp));
            beta = 0;
        end
    end
    
    
    Bp = Bp - beta * (update);
    
    fprintf('    beta = %e, res = %e\n',beta,res_fn_norm(Bp));
    
    div = [Dx,Dy,Dz]*Bp; div(index_gh) = 0; div(index_z1) = 0;
    div(index_x0) = 0; div(index_x1) = 0; div(index_y0) = 0; div(index_y1) = 0;
% Use gmres if lap is large, and direct backslash if lap is small...
%     [phi,flagp,relresp,iterp,resvecp] = gmres(lap,div,1e2,1e-4,1e1);
    phi = lap\div;
%    keyboard();

    phiup = [Dx;Dy;Dz]*phi;
    Bp = Bp-phiup;
    fprintf('    final residual: %e\n',res_fn_norm(Bp));
    
    rs = [rs;res_fn_norm(Bp)];
    Bfullset(:,iters+1) = Bp;
end



% construct laplacian for divergence removal.
% Note that using Dx*Dx instead of Dxx is actually better for this specific
% purpose.


Bp_unmodified = Bp;

rs = rs(2:end);
end

