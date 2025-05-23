(define (problem hook-reach-0)
  (:domain workspace)
  (:objects
    hook - movable
    blue_box - movable
  )
  (:init
    (on hook table)
    (on blue_box table)
    (handempty)
    (clear blue_box)
    (clear hook)
    (inworkspace hook)
    ;; NOTE: blue_box is not in workspace initially
  )
  (:goal (and
    (inhand blue_box)
  ))
)