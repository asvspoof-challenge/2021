function logprob = lgmmprob(data, mu, sigma, w)
% compute the log probability of observations given the GMM
ndim = size(data, 1);
C = sum(mu.*mu./sigma) + sum(log(sigma));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
logprob = -0.5 * (bsxfun(@plus, C',  D));
logprob = bsxfun(@plus, logprob, log(w));