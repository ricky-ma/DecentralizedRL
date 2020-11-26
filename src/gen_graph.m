function A = gen_graph( g_type, N , p )
% generate the doubly stochastic matrix needed
% written on 05.15.2018

A = zeros(N);
if strcmp( g_type , 'ER' )
    % generate the graph
    A = (rand(N) <= p); A = triu(A,1); A = A + A' + eye(N); 
    % turn it into doubly stochastic
    deg_vec = A*ones(N,1); 
    A = A .* ( min( 1./repmat(deg_vec,1,N), 1./repmat(deg_vec',N,1) ) );
    A = A - diag(diag(A)); A = A + diag( ones(N,1) - A*ones(N,1) );
elseif strcmp( g_type, 'ring' )
    A(N+1:(N+1):end) = 1; A(N,1) = 1;
    A = A + A' + eye(N);
    A = A > 0;
    deg_vec = A*ones(N,1); 
    A = A .* ( min( 1./repmat(deg_vec,1,N), 1./repmat(deg_vec',N,1) ) );
    A = A - diag(diag(A)); A = A + diag( ones(N,1) - A*ones(N,1) );
end