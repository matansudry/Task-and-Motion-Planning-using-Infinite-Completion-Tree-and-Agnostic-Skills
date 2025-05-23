(define (problem constrained-packing-0)
	(:domain workspace)
	(:objects
		rack - unmovable
		red_box - box
		yellow_box - box
		cyan_box - box
		blue_box - box
	)
	(:init
		(on red_box table)
		(on yellow_box table)
		(on cyan_box yellow_box)
		(on blue_box table)
		(handempty)
		(clear red_box)
		(clear cyan_box)
		(clear blue_box)
	)
	(:goal (and 
			(on blue_box rack)
		)
	)
)
