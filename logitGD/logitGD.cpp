# include <RcppArmadillo.h>
using namespace Rcpp; 

// [[Rcpp::depends(RcppArmadillo)]]

arma::vec sigmoid(arma::vec x){ // Función sigmoide con entrada de vectores
   int N = x.n_rows ;
   arma::vec s = arma::zeros(N) ;
   
   for (int i = 0; i < N; i++){
      s(i) = 1/(1+std::exp(-x(i))) ;
   }
   
   return s ;
}

arma::vec log(arma::vec x){ // Función log con entrada de vectores
   int N = x.n_rows ;
   arma::vec s = arma::zeros(N) ;
   
   for (int i = 0; i < N; i++){
      s(i) = std::log(x(i)) ;
   }
   
   return s ;
}

// [[Rcpp::export]]
arma::vec predict_logit(arma::vec beta, arma::mat X){ // Función para categorizar con resultados logit
   
   int N = X.n_rows ;
   arma::vec Y_prediction = arma::zeros(N) ;
   
   arma::vec A = sigmoid(X * beta) ;
   
   for (int i = 0; i < N; i++){
      Y_prediction(i) = A(i) > .5 ;
   }
   
   return Y_prediction ;
   
}

// [[Rcpp::export]]
Rcpp::List logitGD(arma::vec Y, arma::mat X, arma::vec init_val, arma::vec Ytest, arma::mat Xtest,
                   double learning_rate = .01, int max_iter = 15000, double tol = 1e-15, bool print_iter = true){
   // Estimación de modelo logit por método del descenso del gradiente
   
   // Tamaño muestral
   double N = Y.n_rows ;
   
   // Pasa el valor inicial al vector de coeficientes
   arma::vec beta = init_val ;
   
   // Computa función de activación
   arma::vec A = sigmoid(X * beta) ;
   
   // Computa función de pérdida inicial
   double loss = arma::as_scalar(arma::sum(Y % log(A) + (1-Y) % log(1-A))) / (-N) ;
   
   // Loop para descenso del gradiente (GD)
   int i = 1 ;
   double loss_new  = 0.0 ;
   double loss_diff = 1e5 ;
   arma::vec grad   = X.t() * (A - Y) ; 
   
   while (i <= max_iter && loss_diff > tol){
      
      // Actualiza beta
      beta = beta - (learning_rate / N) * grad ;
      
      // Recalcula la matriz A y gradiente
      A    = sigmoid(X * beta) ;
      grad = X.t() * (A - Y) ;
      
      // Recalcula función de pérdida
      loss_new  = arma::as_scalar(arma::sum(Y % log(A) + (1-Y) % log(1-A))) / (-N) ;
      loss_diff = std::abs(loss_new - loss) ;
      loss      = loss_new ;
      
      if (i % 500 == 0 && print_iter){
         Rcpp::Rcout << "Iter: " << i << " , Loss: " << loss << " , Loss Diff: " << loss_diff << std::endl ;   
      }
      
      i = i + 1 ;
      
   }
   
   if (i == max_iter+1){
      Rcpp::Rcout << "-- Maximum of iterations reached! --" << std::endl ;
   }
   
   // Predicciones
   arma::vec Y_hat_train = predict_logit(beta, X) ;
   arma::vec Y_hat_test  = predict_logit(beta, Xtest) ;
   
   // Precisión
   double prec_train = 100 - arma::sum(arma::abs(Y_hat_train - Y)) * 100 / N ;
   double prec_test  = 100 - arma::sum(arma::abs(Y_hat_test - Ytest)) * 100 / N ;
   
   Rcpp::Rcout << "*********************************************************" << std::endl ;
   Rcpp::Rcout << "Train Accuracy : " << prec_train << std::endl ;
   Rcpp::Rcout << "Test Accuracy  : " << prec_test << std::endl ;
   
   return List::create(
      _["beta"] = beta,
      _["pred_train"] = Y_hat_train,
      _["pred_test"] = Y_hat_test,
      _["prec_train"] = prec_train,
      _["prec_test"] = prec_test,
      _["loss"] = loss,
      _["iters"] = i - 1
      
   );
   
}

