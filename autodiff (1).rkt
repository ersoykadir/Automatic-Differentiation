#lang racket

;; given
(struct num (value grad)
    #:property prop:custom-write
    (lambda (num port write?)
        (fprintf port (if write? "(num ~s ~s)" "(num ~a ~a)")
            (num-value num) (num-grad num))))
;; given
; (define relu (lambda (x) (if (> (num-value x) 0) x (num 0.0 0.0))))
;; given
; (define mse (lambda (x y) (mul (sub x y) (sub x y))))
