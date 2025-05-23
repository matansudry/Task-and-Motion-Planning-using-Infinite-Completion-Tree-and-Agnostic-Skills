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
		(on yellow_box table)
		(on cyan_box yellow_box)
		(on blue_box table)
		(on red_box blue_box)
		(handempty)
		(clear red_box)
		(clear cyan_box)
	)
	(:goal (and 
			(on blue_box rack)
		)
	)
)
