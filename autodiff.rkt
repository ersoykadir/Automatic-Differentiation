; kadir ersoy
; 2018400252
; compiling: yes
; complete: yes

#lang racket
(provide (all-defined-out))
;; given
(struct num (value grad)
    #:property prop:custom-write
    (lambda (num port write?)
        (fprintf port (if write? "(num ~s ~s)" "(num ~a ~a)")
            (num-value num) (num-grad num))))
;; given
(define relu (lambda (x) (if (> (num-value x) 0) x (num 0.0 0.0))))
;; given
(define mse (lambda (x y) (mul (sub x y) (sub x y))))
(define (get-value num-list)
    (if (list? num-list) 
        (map num-value num-list) (num-value num-list)
    ))
(define (get-grad num-list)
    (if (list? num-list) 
        (map num-grad num-list) (num-grad num-list)
    ))
(define (sub num1 num2)
    (num (- (num-value num1) (num-value num2)) (- (num-grad num1) (num-grad num2)))
    )
(define (add . args)
    (let ((v-list (get-value args)) (g-list (get-grad args)))
    (num (apply + v-list) (apply + g-list))
    ))
(define (listcheck ls);if list return, else make list and return    (kind of typecasting to list)
    (if (list? ls)
        ls
        (list ls)
    )
)
(define (grad-mul args len); rotates the list and calculates the gradient according to the head of the list recursively
    (if (eqv? len 0)
        0
        (+ (foldl * (num-grad(car args)) (listcheck (get-value(cdr args))))       
            (grad-mul 
                (append (listcheck (cdr args)) (listcheck (car args))) 
                (- len 1))
        )
    ))
(define (len list)
    (if (null? list)
        0
        (+ 1 (len (cdr list)))
    )
)
(define (mul . args);
    (let ((v-list (get-value args)));get values
        (num (apply * v-list) ; making a num struct out of multiplication of nums and multiplication of grads(according to chain rule) 
            ;gradient calc 
                ;;get the first num's gradient and multiply it with the values of the rest of the nums
            (+ (foldl * (num-grad(car args)) (listcheck (get-value(cdr args))))    
                ;then rotate left the args list by taking its first element and putting it to the end of the list
                ;do the same gradient calc, stop when the args list returns to its original self after rotations
                (grad-mul 
                    (append (listcheck (cdr args)) (listcheck (car args))) 
                    (- (len args) 1)
                )
            )
        )
    )
)
(define (chash name val var); create num structs, makes gradient 1 if name is equal to var 
    (if (eqv? var name)
        (num val 1.0)
        (num val 0.0)
    ))
(define (varlist lis var);helperfunction--> creates a list by repeating the given variable for length of the given list times
    (if (null? lis)
        '()
        (append (list var) (varlist (cdr lis) var))
    )
)
(define (create-hash names values var);; creates num structs from given values, hashes the names to those num structs
    (let ((ls (map chash names values (varlist names var))))
        (make-hash (map cons names ls))
    )        
)
(define (parse hash expr)
    (cond 
        ((null? expr) '()) 
        ((list? expr) (cons (parse hash (car expr)) (parse hash (cdr expr)) ))
        ((eqv? '+ expr) 'add )
        ((eqv? '* expr) 'mul)
        ((eqv? '- expr) 'sub)
        ((eqv? 'relu expr) 'relu)
        ((eqv? 'mse expr) 'mse)
        ((number? expr) (num expr 0.0))
        (else (hash-ref hash expr))
    )
)
(define (grad names values var expr)
    (let ((hash (create-hash names values var)))
        (num-grad (eval (parse hash expr)))
    )
)(define (partial-grad names values vars expr)
    (get-partial-grad names values vars expr names)
)
(define (get-partial-grad names values vars expr tempNames);; helper func for calling partial-grad recursively, shrinking the temporary names list
    (if (null? vars)
        (if (null? tempNames) ; since vars is empty the rest of the grads will be zero 
            '()
            (varlist tempNames 0.0)
        )
        (if (eqv? (car tempNames) (car vars)) ;if the head of names list is in vars get its grad, if not put 0 and calculate the rest
            (append (listcheck (grad names values (car vars) expr)) (listcheck (get-partial-grad names values (cdr vars) expr (cdr tempNames) )))   
            (append '(0.0) (listcheck (get-partial-grad names values vars expr (cdr tempNames) )) )
        )
    )
)
(define (gradient-descent names values vars lr expr )
    (map - values (map * (varlist names lr) (partial-grad names values vars expr)) 
    )
)
(define (optimize names values vars lr k expr )
    (if (eqv? k 1)
        (gradient-descent names values vars lr expr )
        (gradient-descent names (optimize names values vars lr (- k 1) expr ) vars lr expr)
    )
)
