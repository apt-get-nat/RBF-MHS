function [r] = resB(Bp,dens,pres, Bx0,By0,Bz0,n, ...
        Dx, Dy, Dz, Dxx, Dyy, Dzz, g, ...
        index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, gamma)
% resB.m
% Author: Nathaniel Mathews
% Input:
%         Bp   - the magnetic field in column vector format, in the order
%                [ Bx ; By ; Bz ]
%         dens - the plasma density
%         pres - the plasma pressure
%         Bx0  - the dirichlet boundary on Bx
%         By0  - the dirichlet boundary on By
%         Bz0  - the dirichlet boundary on Bz
%         n    - the number of nodes
%         Dx   - the differentiation matrix with respect to x
%         Dy   - the differentiation matrix with respect to y
%         Dz   - the differentiation matrix with respect to z
%         Dxx  - the second derivative matrix with respect to x
%         Dyy  - the second derivative matrix with respect to y
%         Dzz  - the second derivative matrix with respect to z
%         g    - the gravitational constant in units that agree with dens
%                and pres
%     index_x0 - a column vector whose entries are the row indices of Bp at
%                the lower x boundary
%     index_x1 - a column vector whose entries are the row indices of Bp at
%                the upper x boundary
%     index_y0 - a column vector whose entries are the row indices of Bp at
%                the lower y boundary
%     index_y1 - a column vector whose entries are the row indices of Bp at
%                the upper y boundary
%     index_z0 - a column vector whose entries are the row indices of Bp at
%                the lower z boundary
%     index_z1 - a column vector whose entries are the row indices of Bp at
%                the upper z boundary
%     index_gh - a column vector whose entries are the row indices of Bp at
%                the ghost nodes below the lower z boundary
%        gamma - the hyperviscocity parameter
%
% Output:
%         r    - the residual vector in the order [ rx ; ry ; rz ]


curler = [sparse(n,n), -Dz, Dy; ...
          Dz, sparse(n,n), -Dx; ...
          -Dy, Dx, sparse(n,n)];
      
R1 = 1/(4*pi)*curler*Bp(1:3*n);
R1x = R1(1:n); R1y = R1(n+1:2*n); R1z = R1(2*n+1:3*n); clear R1;

mat = [  sparse(n,n), -spdiags(R1z,0,n,n),  spdiags(R1y,0,n,n); ...
        spdiags(R1z,0,n,n),   sparse(n,n), -spdiags(R1x,0,n,n); ...
       -spdiags(R1y,0,n,n),  spdiags(R1x,0,n,n),   sparse(n,n); ...
       Dx, Dy, Dz];

% hyperviscocity
mat = mat - gamma*[Dxx+Dyy+Dzz, sparse(n,n), sparse(n,n); ...
                   sparse(n,n), Dxx+Dyy+Dzz, sparse(n,n); ...
                   sparse(n,n), sparse(n,n), Dxx+Dyy+Dzz; ...
                   sparse(n,3*n)];

r = mat*Bp - [Dx*pres;Dy*pres;Dz*pres+g*dens;zeros(n,1)];

r = r(1:3*n);

r(2*n+index_z0) = Bp(2*n+index_z0)-Bz0;

% There are a few different options for top and side boundaries. If these
% are altered, they should also be altered in resBJ.m
switch 2
    case 1% radiative
        r(2*n+index_z1) = Dz(index_z1,:)*Bp(2*n+1:3*n) + Bp(2*n+index_z1);
    case 2% constant DzBz
        r(2*n+index_z1) = Dz(index_z1,:)*Bp(2*n+1:3*n);
end
switch 3
    case 1 % Dperpperp Bperp + Bperp = 0 sides
        r(index_x0) = Dxx(index_x0,:)*Bp(1:n) + Bp(index_x0);
        r(index_x1) = Dxx(index_x1,:)*Bp(1:n) + Bp(index_x1);
        r(n+index_y0) = Dyy(index_y0,:)*Bp(n+1:2*n) + Bp(n+index_y0);
        r(n+index_y1) = Dyy(index_y1,:)*Bp(n+1:2*n) + Bp(n+index_y1);
    case 2 % No flux sides
        r(index_x0) = Bp(index_x0);
        r(index_x1) = Bp(index_x1);
        r(n+index_y0) = Bp(n+index_y0);
        r(n+index_y1) = Bp(n+index_y1);
    case 3 % radial field sides
        r(index_x0) = Bp(index_x0).^2 + Bp(n+index_x0).^2;
        r(index_x1) = Bp(index_x1).^2 + Bp(n+index_x1).^2;
        r(n+index_y0) = Bp(index_y0).^2 + Bp(n+index_y0).^2;
        r(n+index_y1) = Bp(index_y1).^2 + Bp(n+index_y1).^2;
end

r(index_gh) = Bp(index_z0)-Bx0;
r(n+index_gh) = Bp(n+index_z0)-By0;
r(2*n+index_gh) = [Dx(index_z0,:),Dy(index_z0,:),Dz(index_z0,:)]*Bp(1:3*n);

end

